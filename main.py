from ultralytics import YOLO
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os

model_path = 'models/face_landmarker.task'
acne_model = YOLO("skinalyzer2.pt")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_face_result = None

# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))
    global latest_face_result
    latest_face_result = result

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        print("No stream :(")
        exit()

    start_time = time.monotonic()
    last_timestamp_ms = -1

    while (True):
        ret, frame = stream.read()
        if not ret:
            print ("No more stream :(")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic()-start_time)*1000)

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms
        landmarker.detect_async(mp_image, timestamp_ms)

        # Run acne detection on the current webcam frame
        acne_results = acne_model.predict(
            source=frame,
            conf=0.2,
            verbose=False
        )

        # Draw YOLO acne bounding boxes onto the frame
        frame = acne_results[0].plot()

        if latest_face_result and latest_face_result.face_landmarks:
            h, w = frame.shape[:2]

            face_landmarks = latest_face_result.face_landmarks[0]

            x_coords = []
            y_coords = []

            for lm in face_landmarks:
                x_coords.append(int(lm.x * w))
                y_coords.append(int(lm.y * h))

            # Calculate face dimensions
            face_width = max(x_coords) - min(x_coords)
            face_height = max(y_coords) - min(y_coords)

            # MediaPipe landmarks for approximate cheek locations
            left_cheek_lm = face_landmarks[50]
            right_cheek_lm = face_landmarks[280]

            # MediaPipe landmark near the center of the forehead
            forehead_lm = face_landmarks[10]

            # Convert normalized coordinates to pixel coordinates
            left_cheek_x = int(left_cheek_lm.x * w)
            left_cheek_y = int(left_cheek_lm.y * h)

            right_cheek_x = int(right_cheek_lm.x * w)
            right_cheek_y = int(right_cheek_lm.y * h)

            # Convert forehead landmark to pixel coordinates
            forehead_x = int(forehead_lm.x * w)
            forehead_y = int(forehead_lm.y * h)

            cheek_y_offset = int(face_height * 0.07)

            left_cheek_y += cheek_y_offset
            right_cheek_y += cheek_y_offset

            # Move forehead region slightly downward
            forehead_y_offset = int(face_height * 0.04)
            forehead_y += forehead_y_offset

            cheek_width = int(face_width * 0.12)
            cheek_height = int(face_height * 0.15)

            # Forehead region size
            forehead_width = int(face_width * 0.32)
            forehead_height = int(face_height * 0.08)

            # Left cheek boundaries
            left_x1 = max(0, left_cheek_x - cheek_width)
            left_x2 = min(w, left_cheek_x + cheek_width)
            left_y1 = max(0, left_cheek_y - cheek_height)
            left_y2 = min(h, left_cheek_y + cheek_height)

            # Right cheek boundaries
            right_x1 = max(0, right_cheek_x - cheek_width)
            right_x2 = min(w, right_cheek_x + cheek_width)
            right_y1 = max(0, right_cheek_y - cheek_height)
            right_y2 = min(h, right_cheek_y + cheek_height)

            # Forehead boundaries
            forehead_x1 = max(0, forehead_x - forehead_width)
            forehead_x2 = min(w, forehead_x + forehead_width)
            forehead_y1 = max(0, forehead_y - forehead_height)
            forehead_y2 = min(h, forehead_y + forehead_height)

            cv2.rectangle(
                frame,
                (left_x1, left_y1),
                (left_x2, left_y2),
                (255, 0, 0),
                2
            )

            cv2.rectangle(
                frame,
                (right_x1, right_y1),
                (right_x2, right_y2),
                (255, 0, 0),
                2
            )

            cv2.rectangle(
                frame,
                (forehead_x1, forehead_y1),
                (forehead_x2, forehead_y2),
                (255, 0, 0),
                2
            )

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
cv2.destroyAllWindows()