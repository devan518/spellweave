from pathlib import Path
import cv2
import mediapipe as mp
import requests

model_path = Path("gesture_recognizer.task")

if not model_path.exists():
    print("Downloading gesture recognition model...")
    response = requests.get("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task")
    with open(model_path, "wb") as file:
        file.write(response.content)
    print("Model downloaded.")

base_options = mp.tasks.BaseOptions
gesture_recognizer = mp.tasks.vision.GestureRecognizer
gesture_recognizer_options = mp.tasks.vision.GestureRecognizerOptions
vision_running_mode = mp.tasks.vision.RunningMode

options = gesture_recognizer_options(base_options=base_options(model_asset_path=str(model_path)), running_mode=vision_running_mode.VIDEO, num_hands=2)

def get_custom_gesture(hand):
    index_up = hand[8].y < hand[6].y
    middle_up = hand[12].y < hand[10].y
    ring_up = hand[16].y < hand[14].y
    pinky_up = hand[20].y < hand[18].y

    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "rock"

    if index_up and middle_up and ring_up and pinky_up:
        return "paper"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "scissors"

    return None


def gesture_handler():
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise RuntimeError("Could not open the camera.")

    try:
        with gesture_recognizer.create_from_options(options) as recognizer:
            while True:
                ret, frame = capture.read()

                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp = int(cv2.getTickCount()/ cv2.getTickFrequency()* 1000)

                result = recognizer.recognize_for_video(
                    mp_image,
                    timestamp
                )

                gestures = {}

                frame_height, frame_width = frame.shape[:2]

                for i, hand in enumerate(result.hand_landmarks):
                    gesture = get_custom_gesture(hand)
                    hand_name = result.handedness[i][0].category_name
                    gestures[hand_name] = gesture

                    for landmark in hand:
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    wrist = hand[0]

                    cv2.putText(
                        frame,
                        gesture or "unknown",
                        (
                            int(wrist.x * frame_width),
                            int(wrist.y * frame_height) - 20
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield gestures, display_frame

    finally:
        capture.release()
        cv2.destroyAllWindows()
