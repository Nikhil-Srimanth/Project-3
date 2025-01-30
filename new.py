import threading
import os
from geopy.geocoders import Nominatim
import openrouteservice
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.garden.mapview import MapView, MapMarker
from kivy.clock import Clock
from kivy.lang import Builder
from plyer import gps, tts

# Initialize APIs
ORS_API_KEY = "5b3ce3597851110001cf6248c755863f0a3b4435bea13cc11dd995a9"
client = openrouteservice.Client(key=ORS_API_KEY)

Builder.load_string('''
<NavigationApp>:
    orientation: 'vertical'
    Label:
        id: status_label
        text: "Ready for Navigation"
        size_hint: (1, 0.1)
    MapView:
        id: mapview
        lat: 52.5200  # Default Berlin coordinates
        lon: 13.4050
        zoom: 15
''')

class NavigationApp(App):
    current_location = [52.5200, 13.4050]
    current_route = None
    current_step = 0
    route_steps = []

    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.status_label = Label(text="Ready for Navigation", size_hint=(1, 0.1))
        self.layout.add_widget(self.status_label)

        self.map_view = MapView(zoom=15, lat=52.5200, lon=13.4050)
        self.layout.add_widget(self.map_view)
        
        self.user_marker = MapMarker(lat=self.current_location[0], lon=self.current_location[1])
        self.map_view.add_marker(self.user_marker)
        
        # Initialize GPS
        self.initialize_gps()
        
        return self.layout

    def initialize_gps(self):
        try:
            gps.configure(on_location=self.on_gps_location)
            gps.start(minTime=5000, minDistance=10)  # Update every 5 seconds or 10 meters
        except NotImplementedError:
            self.status_label.text = "GPS not supported on this platform"

    def on_gps_location(self, **kwargs):
        self.current_location = [kwargs['lat'], kwargs['lon']]
        Clock.schedule_once(self.update_user_position)

    def update_user_position(self, *args):
        self.user_marker.lat = self.current_location[0]
        self.user_marker.lon = self.current_location[1]
        self.map_view.center_on(self.current_location[0], self.current_location[1])
        
        if self.current_route:
            self.check_progress()

    def check_progress(self):
        if self.current_step < len(self.route_steps):
            next_step = self.route_steps[self.current_step]
            distance = self.calculate_distance(self.current_location, next_step['location'])
            
            if distance < 50:  # 50 meters to next step
                self.give_voice_instruction(next_step['instruction'])
                self.current_step += 1

    def calculate_distance(self, loc1, loc2):
        # Simplified distance calculation (Haversine formula)
        from math import radians, sin, cos, sqrt, atan2
        lat1, lon1 = radians(loc1[0]), radians(loc1[1])
        lat2, lon2 = radians(loc2[0]), radians(loc2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return 6371 * c * 1000  # Distance in meters

    def give_voice_instruction(self, text):
        try:
            tts.speak(text)
        except NotImplementedError:
            print("Text-to-speech not supported on this platform")

    def get_route(self, destination):
        location = Nominatim(user_agent="app").geocode(destination)
        if not location:
            return None
        
        end = (location.longitude, location.latitude)
        route = client.directions(
            coordinates=[self.current_location[::-1], end[::-1]],
            profile="foot-walking",
            format="json",
            instructions=True
        )
        
        self.route_steps = [
            {
                'location': [step['location'][1], step['location'][0]],
                'instruction': step['instruction']
            } 
            for step in route['routes'][0]['segments'][0]['steps']
        ]
        
        return route['routes'][0]['geometry']['coordinates']

    def add_route_to_map(self, route):
        self.map_view.remove_marker(self.user_marker)
        self.map_view.clear_widgets()
        
        # Add user marker
        self.user_marker = MapMarker(lat=self.current_location[0], lon=self.current_location[1])
        self.map_view.add_marker(self.user_marker)
        
        # Add destination marker
        end = route[-1][::-1]
        dest_marker = MapMarker(lat=end[0], lon=end[1])
        self.map_view.add_marker(dest_marker)

        # Add route polyline
        self.map_view.polyline = [(coord[1], coord[0]) for coord in route]
        self.current_route = route

    def navigate_to(self, destination):
        route = self.get_route(destination)
        if route:
            self.add_route_to_map(route)
            self.status_label.text = "Navigation started"
            self.current_step = 0

    def on_stop(self):
        gps.stop()
        super().on_stop()

if __name__ == "__main__":
    NavigationApp().run()