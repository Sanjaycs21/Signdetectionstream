this is my code 
import streamlit as st
from streamlit_lottie import st_lottie
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import time
import datetime
#import winsound
import os
import threading
import pickle


# Mock RandomForestClassifier model (replace this with your actual model)
class MockRandomForestClassifier:
    def __init__(self):
        self.n_features_ = 84  # Replace with the actual number of expected features

    def predict(self, data):
        return ['help' for _ in data]  # Mock predictions

# Your existing code without any changes
folder = "D:/tink/models/model2"

model = MockRandomForestClassifier()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'help', 1: 'none'}

# Create a Streamlit app
st.title("Sign Detection")

# Create a placeholder to display the video feed
video_placeholder = st.empty()


# Function to update the video feed
def update_video_feed():
    try:
        data_aux = []
        x_ = []
        y_ = []

        # Use OpenCV to capture video frames
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

        cap = cv2.VideoCapture('http://213.236.250.78/mjpg/video.mjpg')

        while True:
            ret, frame = cap.read()

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Check if the input data matches the expected features
                if len(data_aux) == model.n_features_:
                    prediction_label = model.predict([np.asarray(data_aux)])[0]
                else:
                    prediction_label = 'None'  # Display "None" when input shape doesn't match

                predicted_character = None

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, prediction_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

                if prediction_label != 'help':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

                if prediction_label == 'help':
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    image_path = os.path.join(folder, f'Image_{current_time}.jpg')
                    cv2.imwrite(image_path, frame)
                    print("Help sign frame captured:", image_path)

                    # Display popup with the captured frame
                    show_popup(image_path)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            video_placeholder.image(img)

    except Exception as e:
        print("An error occurred:", e)

# Function to show the popup with the captured frame
def show_popup(image_path):
    # frequency = 2000
    # duration = 2000

    # winsound.Beep(frequency, duration)
    time.sleep(1)

    st.write("Help Needed")

    img = Image.open(image_path)
    st.image(img)


# Create a button to start the video feed
if st.button("Start"):
    update_video_feed()
