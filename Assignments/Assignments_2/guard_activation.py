import speech_recognition as sr
import time

def listen_for_command():
    """
    Listens for a command from the microphone and returns the recognized text.
    Handles errors gracefully.
    """
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for a command...")

        # Adjust for ambient noise to improve accuracy
        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            # Listen to the microphone input
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

            # Use Google's Web Speech API to recognize the audio
            print("Recognizing...")
            command = recognizer.recognize_google(audio)

            # Convert the command to lowercase for easier matching
            return command.lower()

        except sr.WaitTimeoutError:
            # This error means no speech was detected in the timeout period
            print("No speech detected. Listening again.")
            return None
        except sr.UnknownValueError:
            # This error means the API could not understand the audio
            print("Sorry, I did not understand that. Please try again.")
            return None
        except sr.RequestError as e:
            # This error is for API connection issues (e.g., no internet)
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

# --- Main Program Logic ---

if __name__ == "__main__":
    # Define your activation and deactivation phrases
    ACTIVATION_PHRASE = "watch my room"
    DEACTIVATION_PHRASE = "stand down"

    # This is the "state management" flag for your guard
    is_guard_active = False

    print("AI Guard is running. Say 'watch my room' to activate.")

    # Main loop to continuously listen for commands
    while True:
        # Get command from the microphone
        command = listen_for_command()

        if command:
            print(f"You said: '{command}'") # For debugging

            # Check for activation command
            if ACTIVATION_PHRASE in command and not is_guard_active:
                is_guard_active = True
                print("✅ Guard mode ACTIVATED. The flag is now True.")
                # NEXT STEP: Add code here to turn on the webcam and TTS feedback.

            # Check for deactivation command
            elif DEACTIVATION_PHRASE in command and is_guard_active:
                is_guard_active = False
                print("❌ Guard mode DEACTIVATED. The flag is now False.")
                # NEXT STEP: Add code here to turn off the webcam and TTS feedback.

        # A small delay to prevent high CPU usage
        time.sleep(0.1)
