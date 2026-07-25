from pathlib import Path
import requests
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = Path("gesture_recognizer.task")

if not model_path.exists():
    print("downloading gesture recognition model...")
    response = requests.get("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task")

    with open("gesture_recognizer.task", "wb") as f:
        f.write(response.content)
    print("model downloaded ")

base_options = mp.tasks.BaseOptions
gesture_recognizer = mp.tasks.vision.GestureRecognizer
gesture_recognizer_options = mp.tasks.vision.GestureRecognizerOptions
vision_running_mode = mp.tasks.vision.RunningMode

options = gesture_recognizer_options(
    base_options=base_options(model_asset_path=str(model_path)),
    running_mode=vision_running_mode.VIDEO,
    num_hands=2
)

gestures = ["Thumb_Up", "Thumb_Down", "Victory", "Open_Palm", "Closed_Fist", "Pointing_Up"]

def spell_identifier(left_gesture, right_gesture):
    if left_gesture and right_gesture == "Open_Palm":
        print("Shield")

with gesture_recognizer.create_from_options(options) as recognizer:
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        result = recognizer.recognize_for_video(
            mp_image,
            timestamp
        )

        if result.gestures:
            temp_lookup = {}
            for hand_index, hand_gestures in enumerate(result.gestures):
                gesture = hand_gestures[0].category_name
                hand = result.handedness[hand_index][0].category_name.lower()
                if gesture in gestures:
                    temp_lookup[hand] = gesture

            spell_identifier(
                left_gesture=temp_lookup.get("left"),
                right_gesture=temp_lookup.get("right")
            )

        cv2.imshow("Spellweaver", frame)

        if cv2.waitKey(1) == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()

