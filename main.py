import os
import cv2
from fall_agent import FallAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cap = cv2.VideoCapture('4.mp4')

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


output_filename = 'o4.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

fall_agent = FallAgent()

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream or cannot read frame.")
        break

    prediction, keypoints_frame = fall_agent.check(frame)
    if prediction:
        cv2.putText(keypoints_frame, 'FALL DETECTED', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
    else:
        cv2.putText(keypoints_frame, 'No Fall', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Detection", keypoints_frame)

    # Write the keypoints_frame to the output video
    out.write(keypoints_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting video processing.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()