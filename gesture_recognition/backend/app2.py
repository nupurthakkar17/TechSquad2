from flask import Flask, Response
from flask_cors import CORS
import cv2
import mediapipe as mp
import copy
import itertools
import csv
import numpy as np
import tensorflow as tf
import pyttsx3
import threading

app = Flask(__name__)
CORS(app)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Load gesture classification labels
def load_labels(file_path):
    try:
        with open(file_path, encoding='utf-8-sig') as f:
            labels = csv.reader(f)
            return [row[0] for row in labels]
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return []
    except Exception as e:
        print(f"Error loading labels: {e}")
        return []

# Load labels for keypoint classifier
keypoint_classifier_labels = load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')

# KeyPointClassifier class
class KeyPointClassifier:
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.tflite', num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_tensor_idx = self.input_details[0]['index']
        self.interpreter.set_tensor(input_tensor_idx, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()
        output_tensor_idx = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_tensor_idx)
        return np.argmax(np.squeeze(result))

# Initialize KeyPointClassifier
keypoint_classifier = KeyPointClassifier()

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate
engine.setProperty('volume', 1.0)  # Set volume

# Function to speak text
engine = pyttsx3.init()

engine = pyttsx3.init()

engine = pyttsx3.init()

def speak_threaded(text):
    """Runs the speak function in a new thread to prevent blocking."""
    threading.Thread(target=speak, args=(text,), daemon=True).start()

def speak(text):
    if engine._inLoop:
        engine.endLoop()

    engine.say(text)
    engine.runAndWait()
    engine.stop()
# Preprocess landmarks for classification
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

    for index in range(len(temp_landmark_list)):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list)) or 1  # Prevent division by zero
    return [n / max_value for n in temp_landmark_list]

# Calculate landmark list
def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[int(landmark.x * w), int(landmark.y * h)] for landmark in landmarks.landmark]

# Generate video frames with gesture classification and speech output
def generate_frames():
    cap = cv2.VideoCapture(0)
    last_gesture = None  # Store last detected gesture to prevent repetitive speech
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame
             continue  # Skip this frame and move to the next one

        # Process the frame with Mediapipe Hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw hand landmarks and classify gestures
        if results.multi_hand_landmarks:
            detected_gesture = None  # Reset for new frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Classify gesture
                landmark_list = calc_landmark_list(frame, hand_landmarks)
                processed_landmarks = pre_process_landmark(landmark_list)
                gesture_id = keypoint_classifier(processed_landmarks)

                # Prevent index error
                if 0 <= gesture_id < len(keypoint_classifier_labels):
                    detected_gesture = keypoint_classifier_labels[gesture_id]
                else:
                    detected_gesture = "Unknown"

                # Display gesture label on the frame
                cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert gesture to speech only if it's a new detection
            if detected_gesture and detected_gesture != last_gesture:
                speak_threaded(detected_gesture)  # Run speak() in a separate thread
                last_gesture = detected_gesture

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()  # Ensure camera is released when function exits

@app.route('/')
def index():
    return "Welcome to the Live Finger Gesture Detection with Speech Output!"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=5000, threaded=True)  # Allow concurrent processing
