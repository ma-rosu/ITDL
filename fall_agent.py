import cv2
import numpy as np
from tensorflow.python.keras.saving.save import load_model
import mediapipe as mp


class FallAgent:
    def __init__(self):
        self.model = load_model('fall_detection', compile=False)
        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.actions = np.array(['sitting', 'sittingbed', 'notfalling', 'falling'])
        self.sequence = []
        self.predictions = []
        self.threshold = 0.5
        self.prev_keypoints = None

    def check(self, frame):
        def mediapipe_detection(image, model):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model.process(image)  # Make prediction
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image, results

        def extract_keypoints(results):
            pose = np.array(
                [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
                if results.pose_landmarks else np.zeros(33 * 4)

            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
                if results.face_landmarks else np.zeros(468 * 3)

            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
                if results.left_hand_landmarks else np.zeros(21 * 3)

            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
                if results.right_hand_landmarks else np.zeros(21 * 3)

            return np.concatenate([pose, face, lh, rh])

        def calculate_velocity(current, previous, fps=30):
            if previous is None:
                return np.zeros_like(current)
            velocity = (current - previous) * fps
            return velocity

        def draw_styled_landmarks(image, results):
            # Draw pose connections
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                                                    mp.solutions.drawing_utils.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                                    mp.solutions.drawing_utils.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                                    )
            # Draw face connections
            mp.solutions.drawing_utils.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
                                                    mp.solutions.drawing_utils.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                                    mp.solutions.drawing_utils.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                                    )
            # Draw left hand connections
            mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                                                    mp.solutions.drawing_utils.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                                    mp.solutions.drawing_utils.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                                    )
            # Draw right hand connections
            mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                                                    mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                                    mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                                    )

        image, results = mediapipe_detection(frame, self.holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)

        if keypoints is None:
            return False, image

        velocity = calculate_velocity(np.array(keypoints), self.prev_keypoints)
        self.prev_keypoints = np.array(keypoints)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-90:]

        if len(self.sequence) == 90:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            self.predictions.append(np.argmax(res))

            avg_velocity = np.mean(np.linalg.norm(velocity.reshape(-1, 3), axis=1))
            VELOCITY_THRESHOLD = 0.7

            if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > self.threshold:
                    predicted_action = self.actions[np.argmax(res)]

                    if predicted_action == "falling" and avg_velocity > VELOCITY_THRESHOLD:
                        return True, image
                    else:
                        return False, image
        return False, image