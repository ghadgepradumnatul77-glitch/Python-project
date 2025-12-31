import cv2
import mediapipe as mp
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(static_image_mode=False,
                           max_num_faces=5,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Mouth landmarks
                upper_lip = landmarks.landmark[13]
                lower_lip = landmarks.landmark[14]
                left_corner = landmarks.landmark[61]
                right_corner = landmarks.landmark[291]

                ul = np.array([int(upper_lip.x * w), int(upper_lip.y * h)])
                ll = np.array([int(lower_lip.x * w), int(lower_lip.y * h)])
                lc = np.array([int(left_corner.x * w), int(left_corner.y * h)])
                rc = np.array([int(right_corner.x * w), int(right_corner.y * h)])

                mouth_open = np.linalg.norm(ul - ll)
                mouth_width = np.linalg.norm(lc - rc)

                ratio = mouth_open / mouth_width

                avg_corner_y = (lc[1] + rc[1]) / 2
                sad_condition = (ratio < 0.15) and (avg_corner_y > ul[1] + 5)

                if ratio > 0.3:
                    emotion = "Surprised"
                elif ratio > 0.15:
                    emotion = "Happy"
                elif sad_condition:
                    emotion = "Sad"
                else:
                    emotion = "Neutral"

                # -------------------------
                # FIND EMOTION TEXT LOCATION per FACE
                # -------------------------
                # Use nose (landmark 1) for stable face-position labeling
                nose = landmarks.landmark[1]
                text_x = int(nose.x * w)
                text_y = int(nose.y * h) - 20

                cv2.putText(frame, emotion, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0,255,0), thickness=1, circle_radius=1)
                )

        cv2.imshow("Facial Expression Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()