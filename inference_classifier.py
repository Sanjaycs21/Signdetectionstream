import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time
import datetime
import winsound
import os

folder = "D:/tink/models/model2"
model_dict = pickle.load(open('D:/tink/models/model2/model.p', 'rb'))
model = model_dict['model']
def show_popup(image_path):
    frequency = 2000  # Frequency of the beep sound in Hz
    duration = 2000 
    
    popup = tk.Tk()
    popup.title("Help Needed")
    winsound.Beep(frequency, duration)
    time.sleep(1)
    popup.geometry("600x540")
    
    img = Image.open(image_path)
    img = ImageTk.PhotoImage(img)
    
    label = tk.Label(popup, image=img)
    label.pack()
    
    ok_button = tk.Button(popup, text="OK", command=popup.destroy)
    ok_button.pack()

    popup.mainloop()
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'help', 1:'none'}
while True:
    try:
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
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


            prediction_label = model.predict([np.asarray(data_aux)])[0]  # Assuming model returns label string
            predicted_character = None
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, prediction_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
            if prediction_label !='help':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
                
            if prediction_label == 'help':
                # Construct the image path
                current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                image_path = os.path.join(folder, f'Image_{current_time}.jpg')
                
                # Save the image using cv2.imwrite()
                cv2.imwrite(image_path, frame)
                print("Help sign frame captured:", image_path)
                
                # Display popup with the captured frame
                show_popup(image_path)
    except Exception as e:
        print("an error occured:", e)
        continue

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    

cap.release()
cv2.destroyAllWindows()
