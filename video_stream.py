"""
This code was "influenced" by this YOLO-Hand-Detection repo: 
https://github.com/cansik/yolo-hand-detection
"""


import cv2 as cv
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os

abc = "abcdefghiklmnopqrstuvwxy"

def capture_video_stream():
    print("starting webcam...")
    cv.namedWindow("preview")
    vc = cv.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:

        cv.imshow("preview", frame)

        rval, frame = vc.read()

        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv.destroyWindow("preview")
    vc.release()


def media():

    TRAINED_MODEL_PATH = "./resources/trained_model_4.h5"
    if os.path.exists(TRAINED_MODEL_PATH):
        print("Found a backup trained model file, will load now...")
        model = load_model(TRAINED_MODEL_PATH)
        print("Loaded model file:")
    else:
        raise Exception("MOFO!")

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    drawing_styles = mp.solutions.drawing_styles
    cap = cv.VideoCapture(0)
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            image_for_prediction = image.copy()
            
            y , x, c = image.shape
            
            lmxs = []
            lmys = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        lmxs.append(int(lm.x * x))
                        lmys.append(int(lm.y * y))
                
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmark_style(),
                        drawing_styles.get_default_hand_connection_style())
                max_x, max_y, min_x, min_y = max(lmxs), max(lmys), min(lmxs), min(lmys)
                pad_w = int((max_x - min_x) * 0.15)
                pad_h = int((max_y - min_y) * 0.15)
                # cv.putText(image, f"{max_x}, {max_y}, {min_x}, {min_y}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                cv.rectangle(image, (min_x-pad_w, min_y-pad_h), (max_x+pad_w, max_y+pad_h), (255,0,0), 4)
            
                cropped = image_for_prediction[min_y-pad_h:max_y+pad_h, min_x-pad_w:max_x+pad_w]
                # image[0:cropped.shape[0], 0:cropped.shape[1]] = cropped
                s = 30
                if max_x - min_x >= max_y - min_y:
                    dim = (s, int(s * ((max_y - min_y)/(max_x - min_x))))
                else:
                    dim = (int(s * ((max_x - min_x)/(max_y - min_y))), s)
                resized = cv.resize(cropped, dim, interpolation=cv.INTER_AREA)
                temp = np.zeros((s, s, 3)).astype(np.float32)
                # temp[:, :, :] = (255, 255, 255)

                # hpad, wpad = s-dim[0], s-dim[1]
                # print(hpad, wpad)

                # temp[int(hpad/2):dim[1]+int(hpad/2), int(wpad/2):dim[0]+int(wpad/2)] = resized
                temp[0:dim[1], 0:dim[0]] = resized
                gray_temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
                normalized = gray_temp.astype(np.int) / 255
            
                temp1 = np.zeros((s, s, 3))
                temp1[0:s, 0:s, 0] = gray_temp
                temp1[0:s, 0:s, 1] = gray_temp
                temp1[0:s, 0:s, 2] = gray_temp
                image[0:s, 0:s] = temp1

                
                result = np.argmax(model.predict(normalized.reshape(-1, s, s, 1)), axis=-1)
                cv.putText(image, f"result: {abc[int(result)]}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv.imshow('MediaPipe Hands', image)
            if cv.waitKey(5) & 0xFF == 27:
                break            
    cap.release()



if __name__ == "__main__":
    # capture_video_stream()
    media()
