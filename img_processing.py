import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib as plt
from keras.models import Model, load_model
from typing import NamedTuple, Tuple


def load_image(file_path: str) -> np.ndarray:
    return cv.imread(file_path)


def recognize_hand_in_image(img: np.ndarray, should_flip_horizontal: bool, static_image_mode: bool = True, max_num_hands: int = 1, hands_model: mp.solutions.hands.Hands = None) -> NamedTuple or None:
    img = img.copy()
    if should_flip_horizontal:
        img = cv.flip(img, 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img.flags.writeable = False
    if not hands_model:
        with mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=static_image_mode, max_num_hands=max_num_hands) as hands:
            return hands.process(img)
    return hands_model.process(img)


def draw_hand(img:np.ndarray, hands: NamedTuple) -> np.ndarray:
    """hands: (landmarks, handness)"""
    img = img.copy()
    mp_drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    if hands.multi_hand_landmarks:
        for hand_landmarks in hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                        drawing_styles.get_default_hand_landmarks_style(),
                                        drawing_styles.get_default_hand_connections_style())
    return img


def get_hand_frame(img: np.ndarray, hands: NamedTuple, padding: float = 0.2) -> Tuple[tuple] or None:
    h, w, _ = img.shape
    h_landmarks = []
    w_landmarks = []
    result = None
    if hands.multi_hand_landmarks:
        for hand_landmarks in hands.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                w_landmarks.append(int(lm.x * w))
                h_landmarks.append(int(lm.y * h))
        max_x, max_y, min_x, min_y = max(w_landmarks), max(h_landmarks), min(w_landmarks), min(h_landmarks)
        pad_w = int((max_x - min_x) * padding)
        pad_h = int((max_y - min_y) * padding)
        result = (min_x-pad_w, min_y-pad_h), (max_x+pad_w, max_y+pad_h)
    return result


def draw_frame(img, top_left_point: tuple, bottom_right_point: tuple, border_color: tuple = (255, 0, 0), border_width: int = 4) -> np.ndarray:
    img = img.copy()
    cv.rectangle(img, top_left_point, bottom_right_point, border_color, border_width)
    return img


def crop_hand(img, top_left_point: tuple, bottom_right_point: tuple, crop_size: int):
    img = img.copy()
    top_left_x = top_left_point[0]
    top_left_y = top_left_point[1]
    bottom_right_x = bottom_right_point[0]
    bottom_right_y = bottom_right_point[1]

    cropped = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Keeping aspect ratio of the image according to max(width,height)
    w = bottom_right_x - top_left_x
    h = bottom_right_y - top_left_y
    if w >= h:
        dim = (crop_size, int(crop_size * (h/w)))
    else:
        dim = (int(crop_size * (w/h)), crop_size)

    result = np.zeros((crop_size, crop_size, 3)).astype(np.float32)
    if cropped.any():
        resized = cv.resize(cropped, dim, interpolation=cv.INTER_AREA)
        result[0:dim[1], 0:dim[0]] = resized
    return result


def bgr_to_grayscale(img: np.ndarray, is_3d: bool) -> np.ndarray:
    img = img.copy()
    y, x, _ = img.shape
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if not is_3d:
        result = gray
    else:
        gray_3d = np.zeros((y, x, 3))
        gray_3d[0:y, 0:x, 0] = gray
        gray_3d[0:y, 0:x, 1] = gray
        gray_3d[0:y, 0:x, 2] = gray
        result = gray_3d
    return result


def pre_processing(img: np.ndarray, size: int) -> np.ndarray:
    img = bgr_to_grayscale(img, is_3d=False)
    normalized = img.astype(np.int) / 255
    return normalized.reshape(-1, size, size, 1)


def predict(model: Model, img: np.ndarray, size: int) -> np.array:
    img = pre_processing(img, size)
    return np.argmax(model.predict(img), axis=-1)


def plot_image(img: np.ndarray, cmap: str = "gray", figsize: Tuple[int] = (2,2)) -> None:
    img = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize = figsize)
    plt.imshow(img, cmap=cmap)


def load_model_from_file(model_version: int) -> Model:
    model_file_path = f"./resources/trained_model_{model_version}.h5"
    if os.path.exists(model_file_path):
        print("Found a backup trained model file, will load now...")
        model = load_model(model_file_path)
        print("Loaded model file:")
        return model
    else:
        raise Exception(f"Could not find model file at {model_file_path}")
