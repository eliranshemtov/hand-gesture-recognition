"""
This code was "influenced" by this YOLO-Hand-Detection repo: 
https://github.com/cansik/yolo-hand-detection
"""


import cv2 as cv

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


if __name__ == "__main__":
    capture_video_stream()