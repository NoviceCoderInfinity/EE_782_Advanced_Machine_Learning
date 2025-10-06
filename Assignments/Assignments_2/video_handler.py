# video_handler.py
import cv2

def start_webcam():
    """Initializes and returns the webcam capture object."""
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return None
    return cap

def display_frame(cap):
    """Captures and displays a single frame from the webcam."""
    ret, frame = cap.read()
    if ret:
        # Future face recognition logic will go here
        cv2.imshow("AI Guard Feed", frame)
    else:
        print("Error: Failed to capture frame from camera.")
        return False
    return True

def stop_webcam(cap):
    """Releases the webcam and closes all OpenCV windows."""
    print("Stopping webcam...")
    if cap:
        cap.release()
    cv2.destroyAllWindows()
