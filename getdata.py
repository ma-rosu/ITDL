import cv2
import numpy as np
import os
import mediapipe as mp


# Initialize mediapipe solutions for holistic keypoint detection
mp_holistic = mp.solutions.holistic  # Holistic model for pose, face, and hand landmarks
mp_face_mesh = mp.solutions.face_mesh  # Face mesh model
mp_drawing = mp.solutions.drawing_utils  # Utilities for drawing landmarks on images


# Path for data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect (these define the output classes of the model)
actions = np.array(['sitting', 'sittingbed', 'notfalling', 'falling'])

# Number of sequences (videos) expected for each action during data loading
no_sequences = 60

# Each sequence (video) is composed of this many frames (keypoint sets)
sequence_length = 90

# Starting folder number for data collection (useful if you want to resume collection or add more data)
start_folder = 1


def mediapipe_detection(image, model):
    """
    Detects landmarks using mediapipe

    Args:
        image (np.array): The input image in BGR format (from OpenCV).
        model (mediapipe.solutions.holistic.Holistic): The initialized mediapipe Holistic model.

    Returns:
        tuple: image (BGR), mediapipe results
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image color from BGR (OpenCV default) to RGB (mediapipe requirement)
    results = model.process(image)  # Perform the landmark detection on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert image back to BGR for OpenCV display
    return image, results


def draw_styled_landmarks(image, results):
    """
    Draws detected mediapipe landmarks on the image

    Args:
        image: The image on which to draw.
        results: Results from mediapipe detection.
    """
    # Draw face connections with specific colors and thicknesses
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), # Landmark color and size
                             mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1) # Connection line color and size
                             )
    # Draw pose connections with specific colors and thicknesses
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections with specific colors and thicknesses
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections with specific colors and thicknesses
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                             )


def extract_keypoints(results):
    """
    Extracts x, y, z coordinates and visibility of detected landmarks
    and flattens them into a single numpy array.

    Args:
        results: Results from mediapipe detection.

    Returns:
        np.array: A concatenated numpy array of all extracted keypoints.
    """
    # Extract pose landmarks (x, y, z, visibility). If no pose, return array of zeros.
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4) # 33 landmarks * 4 values (x,y,z,visibility)

    # Extract face landmarks (x, y, z). If no face, return array of zeros.
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3) # 468 landmarks * 3 values (x,y,z)

    # Extract left hand landmarks (x, y, z). If no left hand, return array of zeros.
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3) # 21 landmarks * 3 values (x,y,z)

    # Extract right hand landmarks (x, y, z). If no right hand, return array of zeros.
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3) # 21 landmarks * 3 values (x,y,z)

    # Concatenate keypoints into a single array
    return np.concatenate([pose, face, lh, rh])


def set_folders():
    """
    Creates folders.
    """
    # Create the base data directory if it doesn't exist
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    # Iterate through each defined action
    for action in actions:
        # Loop through the desired number of sequences for each action
        for sequence in range(1, no_sequences + 1):
            try:
                # Create a subfolder for each action and sequence
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                # If folder already exists, pass
                pass


def get_frames():
    """
    Gets frames from the webcam and processes them with mediapipe.
    """
    # Initialize video capture from the webcam
    cap = cv2.VideoCapture(0)
    # Use mediapipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Loop through each defined action
        for action in actions:
            # Loop through the specified number of sequences for each action
            for sequence in range(start_folder, start_folder + no_sequences): # Allows starting from a specific folder number
                # Loop through each frame for the current sequence
                for frame_num in range(sequence_length):

                    ret, frame = cap.read() # Read a frame from the webcam. ret is True if frame is read successfully.

                    image, results = mediapipe_detection(frame, holistic) # Process the frame with mediapipe

                    draw_styled_landmarks(image, results) # Draw the detected landmarks with custom styles

                    # Display collection status messages on the screen
                    if frame_num == 0: # For the first frame of a new sequence
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200), # Big message for start
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), # Info message
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image) # Show the image
                        cv2.waitKey(5000) # Wait for 5 seconds to give user time to prepare
                    else: # For subsequent frames in a sequence
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), # Info message
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image) # Show the image

                    # Extract keypoints from the mediapipe results
                    keypoints = extract_keypoints(results)
                    # Define the path where the keypoints NumPy array will be saved
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints) # Save the keypoints array to a .npy file

                    # Break the loop if 'q' is pressed (user can quit manually)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        cap.release() # Release the webcam resource
        cv2.destroyAllWindows() # Close all OpenCV windows
