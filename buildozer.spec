[app]

# Title of your application
title = NavigationApp

# Package name (reverse domain style)
package.name = navigationapp

# Package domain (used for the package name)
package.domain = org.test

# Source directory where your main.py is located
source.dir = .

# Source files to include (main.py and any others)
source.include_patterns = main.py, assets/*.png

# Application version
version = 1.0

# Requirements for the app (Python modules)
requirements = 
    python3,
    kivy,
    geopy,
    openrouteservice,
    plyer,
    requests,
    certifi

# Kivy Garden modules
garden_requirements = mapview

# Android specific configurations
android.permissions = 
    INTERNET,
    ACCESS_COARSE_LOCATION,
    ACCESS_FINE_LOCATION,
    ACCESS_BACKGROUND_LOCATION

# Android API level
android.api = 29

# Minimum Android SDK version
android.minapi = 21

# Android SDK version to target
android.sdk = 23

# Android NDK version
android.ndk = 19c

# Architecture (armeabi-v7a is compatible with most devices)
android.arch = armeabi-v7a

# Orientation (portrait or landscape)
orientation = portrait

# Features (enable hardware features)
android.features = 
    android.hardware.location,
    android.hardware.location.gps,
    android.hardware.internet

# Log level (2 for verbose, useful for debugging)
log_level = 2

# Set to debug mode for development
debug = 1

# Presplash screen (optional)
# presplash.filename = presplash.png

# Icon (optional)
# icon.filename = icon.png

# Set the main entry point of the app
entrypoint = main.py

# Prevent NumPy from being included (not needed here)
numpy = omit

# Enable OpenGL ES 2.0
fullscreen = 0
android.allow_backup = True
android.touchscreen = 1
android.opengl = 20