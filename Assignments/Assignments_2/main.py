# main.py
import cv2
import time
from audio_handler import listen_for_command, speak
from video_handler import start_webcam, stop_webcam, display_frame

if __name__ == "__main__":
    # --- Configuration ---
    ACTIVATION_PHRASE = "start"
    DEACTIVATION_PHRASE = "stop"
    
    # --- State Management ---
    is_guard_active = False
    cap = None

    speak("AI Guard is ready.")
    
    while True:
        # Always listen for commands, regardless of guard state
        command = listen_for_command()
        
        if command:
            print(f"🗣️ You said: '{command}'")

            if ACTIVATION_PHRASE in command and not is_guard_active:
                is_guard_active = True
                speak("Guard mode activated.")
                cap = start_webcam()
                if not cap: # Handle camera failure
                    is_guard_active = False

            elif DEACTIVATION_PHRASE in command and is_guard_active:
                is_guard_active = False
                speak("Guard mode deactivated.")
                stop_webcam(cap)
                cap = None
        
        # If guard is active, display the camera feed
        if is_guard_active and cap:
            if not display_frame(cap):
                # Stop if the frame could not be displayed
                is_guard_active = False
                stop_webcam(cap)
                cap = None

        # Check for quit key ('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Quitting program.")
            break
            
    # Clean up resources before exiting
    if cap:
        stop_webcam(cap)
    print("AI Guard has shut down.")
