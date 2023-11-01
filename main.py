import dataclasses

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 as cv
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import math

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def calcDis(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2))

# pA is the center angle.
def calcAngle(dataset, pA, pB, pC):
    pA = dataset.hand_landmarks[0][pA]
    pB = dataset.hand_landmarks[0][pB]
    pC = dataset.hand_landmarks[0][pC]

    a = calcDis(pB.x, pB.y, pB.z, pC.x, pC.y, pC.z)
    b = calcDis(pA.x, pA.y, pA.z, pC.x, pC.y, pC.z)
    c = calcDis(pA.x, pA.y, pA.z, pB.x, pB.y, pB.z)

    cos_angle = (b * b + c * c - a * a) / (2 * b * c)
    angle = np.arccos(cos_angle) / np.pi * 180

    return angle

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv.putText(annotated_image, f"{handedness[0].category_name}",
                   (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                   FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

    return annotated_image


# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       running_mode=mp.tasks.vision.RunningMode.IMAGE,
                                       num_hands=1,
                                       min_hand_detection_confidence=0.5,
                                       min_hand_presence_confidence=0.5,
                                       min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

# frame = mp.Image.create_from_file("image.jpg")
# print(type(frame))

# Initialize the camera
cap = cv.VideoCapture(1)
# cap.set(cv.CAP_PROP_FPS, 30)
# print(cap.get(cv.CAP_PROP_FPS))

if not cap.isOpened():
    print("Failed to open the camera.")
    exit()
cv.namedWindow("Frame", cv.WINDOW_NORMAL)

t0 = time.time()

# Define a motion - Range 1->...->1
high_angle = 60
low_angle = 35

reachEnd = 1

prev_angle = 45
cur_angle = 45
cnt = 0

isWorking = 0

# Counter
perfect_count = 0
good_count = 0
bad_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive the image.")
        break

    frame = cv.resize(frame, (400, int(frame.shape[0] * 400 / frame.shape[1])))
    cv.resizeWindow("Frame", 400, int(frame.shape[0] * 400 / frame.shape[1]))
    frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
    detection_result = detector.detect(frame)
    annotated_image = draw_landmarks_on_image(frame.numpy_view(), detection_result)

    # Text the size of the image
    # txt = "Image Size: " + "{:.2f}".format(annotated_image.size / 1024 / 1024) + " MB"
    # cv.putText(annotated_image, txt, (10, 290), cv.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)

    if len(detection_result.hand_landmarks) == 1:
        prev_angle = cur_angle
        cur_angle = calcAngle(detection_result, 0, 4, 20)
        txt2 = "" + "{:.2f}".format(cur_angle)

        # This section is for checking whether it is valid.
        if (calcAngle(detection_result, 6, 5, 8) > 160
        and calcAngle(detection_result, 10, 12, 9) > 160
        and calcAngle(detection_result, 14, 16, 13) > 160
        and calcAngle(detection_result, 18, 20, 17) > 160):
            nwIsWorking = 1
        else:
            nwIsWorking = 0

        if nwIsWorking != isWorking and nwIsWorking == 0:
            print("It seems that you have stopped doing this motion.")

        isWorking = nwIsWorking
        if isWorking:
            if cur_angle >= high_angle:
                if reachEnd == 2:
                    bad_count += 1
                    print("This is not a perfect motion.")
                elif reachEnd == 3:
                    perfect_count += 1
                    print("Motion detected.")
                reachEnd = 1
            elif low_angle <= cur_angle < high_angle:
                reachEnd = max(reachEnd, 2)
            else:
                reachEnd = max(reachEnd, 3)
    else:
        txt2 = "No hands detected."
    # cv.putText(annotated_image, txt2, (0, 100), cv.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)

    txt_perfect_motions = "Perfect Motions Count: " + str(perfect_count)
    txt_bad_motions = "Bad Motions Count: " + str(bad_count)

    cv.putText(annotated_image, txt_perfect_motions, (0, 260),
               cv.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(annotated_image, txt_bad_motions, (0, 280),
               cv.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)

    cv.imshow("Frame", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
    cv.imshow("Frame", cv.cvtColor(annotated_image, cv.COLOR_RGBA2RGB))

    if cv.waitKey(1) == ord('q'):
        break

t1 = time.time()

# print(t1-t0)

cv.destroyAllWindows()