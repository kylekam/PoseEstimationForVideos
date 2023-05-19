import mediapipe as mp
import cv2
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

VIDEO_FILENAMES = [""]
VIDEO_FILENAME = "2YT8V.mp4"
VIDEO_PATH = "./Charades_v1_480/"
pose = mp_pose.Pose(min_detection_confidence = 0.5,
                    min_tracking_confidence = 0.5)


cap = cv2.VideoCapture(VIDEO_PATH + VIDEO_FILENAME)
if (cap.isOpened() == False):
    print("Error opening " + VIDEO_FILENAME)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output = cv2.VideoWriter("output.mp4", 0x7634706d,
                         fps, (frame_width,frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret):
        results = pose.process(frame)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        output.write(frame)
        cv2.imshow("output", frame)
    else:
        break

cv2.destroyAllWindows()
output.release()
cap.release()