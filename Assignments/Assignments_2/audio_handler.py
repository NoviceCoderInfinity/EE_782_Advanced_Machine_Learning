# audio_handler.py
import speech_recognition as sr
import pyttsx3

# --- Text-to-Speech Setup ---
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    tts_engine = None

def speak(text):
    """Uses the TTS engine to speak the given text."""
    print(f"🤖 AI Guard says: {text}")
    if tts_engine:
        tts_engine.say(text)
        tts_engine.runAndWait()

# --- Speech Recognition Function ---
def listen_for_command():
    """Listens for a command and returns the recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("Recognizing...")
            command = recognizer.recognize_google(audio)
            return command.lower()
        except sr.WaitTimeoutError:
            return None # Don't print anything, main loop will handle it
        except sr.UnknownValueError:
            speak("Sorry, I did not understand that.")
            return None
        except sr.RequestError as e:
            speak("Could not connect to speech service.")
            print(f"Error: {e}")
            return None
