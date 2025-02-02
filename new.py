import cv2
import numpy as np
from ultralytics import YOLO
import pvporcupine
import pyaudio
import speech_recognition as sr
import struct
import pyttsx3
import threading
from threading import Event, Lock
import geocoder
import folium
from geopy.geocoders import Nominatim
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.label import Label
import openrouteservice
from kivy.garden.mapview import MapView, MapMarker
from kivy.lang import Builder
from plyer import gps
from kivy.core.window import Window
import queue
import time
from kivy.graphics.texture import Texture
from math import radians, sin, cos, sqrt, atan2
import os

# API Keys
ORS_API_KEY = "5b3ce3597851110001cf6248c755863f0a3b4435bea13cc11dd995a9"
PORCUPINE_ACCESS_KEY = "YluWAVPbPIxSkJtljSpqvkwYjv72DjE5yn02NorEaVjtKzio8OspNA=="

# Kivy UI Layout
kv = '''
<WakeScreen>:
    BoxLayout:
        orientation: 'vertical'
        spacing: 10
        padding: 10
        
        Label:
            text: "Voice Assistant"
            font_size: '30sp'
            size_hint_y: 0.3
            
        Label:
            id: status_label
            text: "Listening for wake word..."
            size_hint_y: 0.3
            
        Label:
            id: command_label
            text: ""
            size_hint_y: 0.3
            
        Button:
            text: "Exit Application"
            size_hint_y: 0.1
            on_press: app.stop()

<ObjectDetectionScreen>:
    BoxLayout:
        orientation: 'vertical'
        spacing: 10
        padding: 10
        
        Image:
            id: camera_preview
            size_hint_y: 0.7
            
        Label:
            id: detection_label
            text: "Object Detection Active"
            size_hint_y: 0.2
            
        Button:
            text: "Back to Main"
            size_hint_y: 0.1
            on_press: root.go_back()
            
<NavigationScreen>:
    BoxLayout:
        orientation: 'vertical'
        spacing: 10
        padding: 10
        
        MapView:
            id: mapview
            lat: 52.5200 
            lon: 13.4050
            zoom: 15
            size_hint_y: 0.7
            
        Label:
            id: nav_status
            text: "Navigation Active"
            size_hint_y: 0.2
            
        Button:
            text: "Back to Main"
            size_hint_y: 0.1
            on_press: root.go_back()
'''

class SpeechManager:
    def __init__(self):
        # Initialize text-to-speech engine
        self.tts_engine = None
        self.initialize_tts()
        
        # Setup threading components
        self.tts_queue = queue.Queue()
        self.tts_thread_running = True
        self.tts_lock = Lock()
        
        # Start TTS worker thread
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

    def initialize_tts(self):
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 1.0)  # Volume level
        except Exception as e:
            print(f"TTS initialization error: {e}")

    def _tts_worker(self):
        while self.tts_thread_running:
            try:
                # Get text from queue with timeout
                text = self.tts_queue.get(timeout=1)
                if self.tts_engine:
                    with self.tts_lock:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")
                self.initialize_tts()

    def speak(self, text):
        print(f"Speaking: {text}")  # Debug output
        self.tts_queue.put(text)

    def cleanup(self):
        self.tts_thread_running = False
        if self.tts_thread.is_alive():
            self.tts_thread.join(timeout=1)

class WakeScreen(Screen):
    def __init__(self, **kwargs):
        super(WakeScreen, self).__init__(**kwargs)
        self.porcupine = None
        self.audio_stream = None
        self.pa = None
        self.running = True
        self.wake_thread = None
    def set_speech_manager(self, speech_manager):
        self.speech_manager = speech_manager
    def on_enter(self):
        self.running = True
        self.start_wake_detection()

    def on_leave(self):
        self.running = False
        self.cleanup_audio()

    def cleanup_audio(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()

    def start_wake_detection(self):
        try:
            # Initialize Porcupine wake word detector
            self.porcupine = pvporcupine.create(
                access_key=PORCUPINE_ACCESS_KEY,
                keyword_paths=["optic_en_windows_v3_0_0.ppn"],
                sensitivities=[0.5]
            )
            
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            # Start wake word detection thread
            self.wake_thread = threading.Thread(target=self.wake_detection_loop, daemon=True)
            self.wake_thread.start()
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            if self.speech_manager:
                self.speech_manager.speak("Error initializing audio system")

    def wake_detection_loop(self):
        while self.running:
            try:
                pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                if self.porcupine.process(pcm) >= 0:
                    Clock.schedule_once(lambda dt: self.handle_wake_word())
                    
            except Exception as e:
                print(f"Error in wake detection: {e}")
                break

    def handle_wake_word(self):
        if self.speech_manager:
            self.speech_manager.speak("How can I help you?")
            Clock.schedule_once(lambda dt: self.listen_for_command())

    def listen_for_command(self, dt=None):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("Listening for command...")
                self.ids.status_label.text = "Listening for command..."
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")
                self.ids.command_label.text = f"Command: {command}"
                self.process_command(command)
                
        except sr.UnknownValueError:
            if self.speech_manager:
                self.speech_manager.speak("I couldn't understand you.")
            self.ids.status_label.text = "Couldn't understand command"
            
        except Exception as e:
            print(f"Command recognition error: {e}")
            if self.speech_manager:
                self.speech_manager.speak("There was an error processing your command.")
            self.ids.status_label.text = "Error processing command"
    def process_command(self, command):
        if ("detect" in command and ("object" in command or "objects" in command)) or \
           ("describe" in command and ("environment" in command or "surroundings" in command)):
            if self.speech_manager:
                self.speech_manager.speak("Starting object detection mode. Say 'detect' to detect objects or 'back to main' to return.")
            Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'object'))
            
        elif "navigate" in command or "navigation" in command:
            if self.speech_manager:
                self.speech_manager.speak("Starting navigation.")
            Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'navigation'))
            
        else:
            if self.speech_manager:
                self.speech_manager.speak("Command not recognized. Please try again.")
            self.ids.status_label.text = "Command not recognized"

class ObjectDetectionScreen(Screen):
    def __init__(self, **kwargs):
        super(ObjectDetectionScreen, self).__init__(**kwargs)
        self.model = None
        self.cap = None
        self.running = True
        self.detection_lock = Lock()
        self.detect_now = False  
        self.detection_thread = None
        self.load_model()
        
    def set_speech_manager(self, speech_manager):
        self.speech_manager = speech_manager
        
    def load_model(self):
        try:
            self.model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")

    def on_enter(self):
        self.running = True
        self.setup_camera()
        if self.cap and self.cap.isOpened():
            Clock.schedule_interval(self.update_camera_preview, 1.0/30.0)
            self.start_voice_command_listener()
        else:
            if self.speech_manager:
                self.speech_manager.speak("Camera not available")

    def start_voice_command_listener(self):
        self.voice_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
        self.voice_thread.start()

    def listen_for_commands(self):
        recognizer = sr.Recognizer()
        while self.running:
            try:
                with sr.Microphone() as source:
                    print("Listening for object detection commands...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    command = recognizer.recognize_google(audio).lower()
                    
                    if "detect" in command or "what do you see" in command:
                        self.detect_now = True
                    elif "back to main" in command or "go back" in command:
                        Clock.schedule_once(lambda dt: self.go_back())
                        break
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print(f"Command recognition error: {e}")
                continue

    def setup_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
        except Exception as e:
            print(f"Camera setup error: {e}")
            self.cap = None

    def on_leave(self):
        self.running = False
        if hasattr(self, 'voice_thread'):
            self.voice_thread.join(timeout=1)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def update_camera_preview(self, dt):
        if not self.cap or not self.cap.isOpened():
            return False

        ret, frame = self.cap.read()
        if not ret:
            return False

        if self.detect_now:
            self.process_frame(frame)
            self.detect_now = False

        buf = cv2.flip(frame, 0)
        buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
        texture = Texture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.ids.camera_preview.texture = texture
        return True

    def process_frame(self, frame):
        if self.model is None:
            return

        with self.detection_lock:
            try:
                results = self.model(frame, stream=True)
                frame_height, frame_width = frame.shape[:2]
                detected_objects = {}

                for result in results:
                    for box in result.boxes:
                        try:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if conf < 0.5:
                                continue

                            name = self.model.names[cls]
                            center_x = (x1 + x2) / 2

                            position = "in front of you"
                            if center_x < frame_width / 3:
                                position = "on the left"
                            elif center_x > 2 * frame_width / 3:
                                position = "on the right"

                            detected_objects[name] = position

                        except Exception as e:
                            print(f"Error processing detection: {e}")
                            continue

                if self.speech_manager and detected_objects:
                    announcement = "I can see "
                    object_descriptions = [f"a {obj} {pos}" for obj, pos in detected_objects.items()]
                    announcement += ", and ".join(object_descriptions)
                    self.speech_manager.speak(announcement)

            except Exception as e:
                print(f"Detection error: {e}")

    def go_back(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.manager.current = 'wake'

class NavigationScreen(Screen):
    def __init__(self, **kwargs):
        super(NavigationScreen, self).__init__(**kwargs)
        self.current_location = [81.0545, 16.3437] 
        self.current_route = None
        self.current_step = 0
        self.route_steps = []
        self.client = openrouteservice.Client(key=ORS_API_KEY)
        self.geolocator = Nominatim(user_agent="blind_assistant")
    def set_speech_manager(self, speech_manager):
        self.speech_manager = speech_manager
    def on_enter(self):
        self.initialize_gps()
        if self.speech_manager:
            self.speech_manager.speak("Please say your destination")
        threading.Thread(target=self.listen_for_destination, daemon=True).start()

    def initialize_gps(self):
        try:
            gps.configure(on_location=self.on_gps_location)
            gps.start(minTime=5000, minDistance=10)
        except Exception as e:
            print(f"GPS initialization error: {e}")
            if self.speech_manager:
                self.speech_manager.speak("GPS not available")

    def on_gps_location(self, **kwargs):
        try:
            self.current_location = [kwargs['lat'], kwargs['lon']]
            Clock.schedule_once(lambda dt: self.update_navigation())
        except Exception as e:
            print(f"GPS update error: {e}")

    def listen_for_destination(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                self.ids.nav_status.text = "Listening for destination..."
                audio = recognizer.listen(source, timeout=10)
                destination = recognizer.recognize_google(audio)
                Clock.schedule_once(lambda dt: self.start_navigation(destination))
        except Exception as e:
            print(f"Error in destination recognition: {e}")
            if self.speech_manager:
                self.speech_manager.speak("Could not understand destination")
            self.ids.nav_status.text = "Could not understand destination"

    def start_navigation(self, destination):
        try:
            location = self.geolocator.geocode(destination)
            if not location:
                if self.speech_manager:
                    self.speech_manager.speak("Could not find that location")
                self.ids.nav_status.text = "Location not found"
                return

            route = self.get_route((location.longitude, location.latitude))
            if route:
                self.add_route_to_map(route)
                if self.speech_manager:
                    self.speech_manager.speak(f"Starting navigation to {destination}")
                self.ids.nav_status.text = f"Navigating to {destination}"
            else:
                if self.speech_manager:
                    self.speech_manager.speak("Could not calculate route")
                self.ids.nav_status.text = "Route calculation failed"
        except Exception as e:
            print(f"Navigation error: {e}")
            if self.speech_manager:
                self.speech_manager.speak("Error starting navigation")
            self.ids.nav_status.text = "Navigation error"

    def get_route(self, destination):
        try:
            coords = [
                (self.current_location[1], self.current_location[0]),  # [lon, lat]
                destination  # [lon, lat]
            ]
            
            route = self.client.directions(
                coordinates=coords,
                profile='foot-walking',
                format='geojson',
                instructions=True
            )
            
            if 'features' in route and len(route['features']) > 0:
                self.route_steps = []
                for step in route['features'][0]['properties']['segments'][0]['steps']:
                    self.route_steps.append({
                        'instruction': step['instruction'],
                        'distance': step['distance'],
                        'location': [step['location'][1], step['location'][0]]  # [lat, lon]
                    })
                return route['features'][0]['geometry']['coordinates']
            return None
            
        except Exception as e:
            print(f"Route calculation error: {e}")
            return None

    def add_route_to_map(self, route_coords):
        try:
            for marker in self.ids.mapview.children[:]:
                if isinstance(marker, MapMarker):
                    self.ids.mapview.remove_marker(marker)
            
            start_marker = MapMarker(lat=self.current_location[0], lon=self.current_location[1])
            self.ids.mapview.add_marker(start_marker)
            
            # Add destination marker
            end_coords = route_coords[-1]
            end_marker = MapMarker(lat=end_coords[1], lon=end_coords[0])
            self.ids.mapview.add_marker(end_marker)
            
            # Store route
            self.current_route = route_coords
            self.current_step = 0
            
            # Center map on start position
            self.ids.mapview.center_on(self.current_location[0], self.current_location[1])
            
        except Exception as e:
            print(f"Error adding route to map: {e}")

    def update_navigation(self):
        if not self.current_route or not self.route_steps:
            return
            
        try:
            # Update user position on map
            self.ids.mapview.center_on(self.current_location[0], self.current_location[1])
            
            # Check if we've reached the next step
            if self.current_step < len(self.route_steps):
                next_step = self.route_steps[self.current_step]
                distance = self.calculate_distance(
                    self.current_location,
                    next_step['location']
                )
                
                # If within 20 meters of next step
                if distance < 20:
                    if self.speech_manager:
                        self.speech_manager.speak(next_step['instruction'])
                    self.current_step += 1
                    
                    # Check if we've reached the destination
                    if self.current_step >= len(self.route_steps):
                        if self.speech_manager:
                            self.speech_manager.speak("You have reached your destination")
                        self.ids.nav_status.text = "Destination reached"
                        
        except Exception as e:
            print(f"Navigation update error: {e}")

    def calculate_distance(self, point1, point2):
        R = 6371000  # Earth's radius in meters
        
        lat1, lon1 = radians(point1[0]), radians(point1[1])
        lat2, lon2 = radians(point2[0]), radians(point2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)*2 + cos(lat1) * cos(lat2) * sin(dlon/2)*2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c

    def go_back(self):
        gps.stop()
        self.manager.current = 'wake'

class BlindAssistantApp(App):
    def build(self):
        Builder.load_string(kv)
        
        self.speech_manager = SpeechManager()
        
        sm = ScreenManager()
        
        wake_screen = WakeScreen(name='wake')
        obj_screen = ObjectDetectionScreen(name='object')
        nav_screen = NavigationScreen(name='navigation')
        
        wake_screen.set_speech_manager(self.speech_manager)
        obj_screen.set_speech_manager(self.speech_manager)
        nav_screen.set_speech_manager(self.speech_manager)

        sm.add_widget(wake_screen)
        sm.add_widget(obj_screen)
        sm.add_widget(nav_screen)
        
        return sm

    def on_stop(self):
        self.speech_manager.cleanup()

if __name__ == "__main__":
    try:
        Window.size = (800, 600)
        BlindAssistantApp().run()
    except Exception as e:
        print(f"Application error: {e}")
