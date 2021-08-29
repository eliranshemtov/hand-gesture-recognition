import cv2 as cv
import mediapipe as mp
from img_processing import recognize_hand_in_image, draw_frame, draw_hand, get_hand_frame, crop_hand,\
    bgr_to_grayscale, predict, load_model_from_file


CLASSES = "ABW"
MODEL_VERSION = 13
MODEL_IMAGE_SIZE = 128
MIN_HANDS_DETECTION_CONFIDENCE = 0.6
MIN_HANDS_TRACKING_CONFIDENCE = 0.6


def main(model_version):
    model = load_model_from_file(model_version)
    cap = cv.VideoCapture(0)
    with mp.solutions.hands.Hands(min_detection_confidence=MIN_HANDS_DETECTION_CONFIDENCE,
                                  min_tracking_confidence=MIN_HANDS_TRACKING_CONFIDENCE, max_num_hands=1) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue
            recognized_hands = recognize_hand_in_image(image, should_flip_horizontal=False, static_image_mode=False,
                                                       max_num_hands=1, hands_model=hands)
            current_frame = draw_hand(image, recognized_hands)
            hands_frame = get_hand_frame(current_frame, recognized_hands)
            if hands_frame:
                current_frame = draw_frame(current_frame, hands_frame[0], hands_frame[1])
                cropped_hand = crop_hand(image, hands_frame[0], hands_frame[1], MODEL_IMAGE_SIZE)
                current_frame[0:MODEL_IMAGE_SIZE, 0:MODEL_IMAGE_SIZE] = bgr_to_grayscale(cropped_hand, is_3d=True)
                result = predict(model, cropped_hand, MODEL_IMAGE_SIZE)
                draw_result(current_frame, result)
            cv.imshow('Sign Language Detector', current_frame)
            if cv.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def draw_result(current_frame, result):
    if 0 <= result < len(CLASSES):
        cv.putText(current_frame, f"Result: {CLASSES[int(result)]}", (10, 70),
                   cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


if __name__ == "__main__":
    main(model_version=MODEL_VERSION)
