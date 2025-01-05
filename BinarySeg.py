#Computer Vision Task
# Abdullah Kamal
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# using YOLOv8 Segmentation Model
yolo_model = YOLO("yolov8n-seg.pt") 

# I am using Mediapipe Pose for Skeleton Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# the 3 minutes video I cut from the orgignal video as required
video_path = "video13rdmin.mp4"  # note that the file is in the same directory if not, you should add the complete path 
cap = cv2.VideoCapture(video_path)

# the expected output video
output_path = "BinaryS_video13rdmin.mp4"
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# looping over frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # running YOLOv8 Segmentation
    results = yolo_model(frame)
    binary_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)  # Initialize binary mask
    for result in results:
        for mask, box, cls in zip(result.masks.data.cpu().numpy(), result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            if int(cls) == 0:  # ID 0 to only detect each 'person'
                # the bounding box:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Human", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # I am writing Human over the box of the detected person becuase there are many objects in each frame 

                # resizing the segmentation mask to match the frame size
                resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

                # using binary mask to have stable lightening 
                binary_resized_mask = (resized_mask > 0.5).astype(np.uint8)

                # combining the binary mask for this person into the global binary mask
                binary_mask = cv2.bitwise_or(binary_mask, binary_resized_mask)

                # Overlay skeleton landmarks
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    results_pose = pose.process(roi_rgb)
                    if results_pose.pose_landmarks:
                        for landmark in results_pose.pose_landmarks.landmark:
                            cx = int(landmark.x * (x2 - x1)) + x1
                            cy = int(landmark.y * (y2 - y1)) + y1
                            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                        for connection in mp_pose.POSE_CONNECTIONS:
                            start_idx, end_idx = connection
                            start = results_pose.pose_landmarks.landmark[start_idx]
                            end = results_pose.pose_landmarks.landmark[end_idx]

                            start_cx = int(start.x * (x2 - x1)) + x1
                            start_cy = int(start.y * (y2 - y1)) + y1
                            end_cx = int(end.x * (x2 - x1)) + x1
                            end_cy = int(end.y * (y2 - y1)) + y1
                            cv2.line(frame, (start_cx, start_cy), (end_cx, end_cy), (255, 0, 0), 2)

    # creating a binary segmentation visualization
    colored_binary_mask = cv2.merge([binary_mask * 255, binary_mask * 255, binary_mask * 255]) 
    frame_with_mask = cv2.addWeighted(frame, 0.7, colored_binary_mask, 0.3, 0)

    # adding the processed frame to the output video
    out.write(frame_with_mask)

# releasing the resources
cap.release()
out.release()
cv2.destroyAllWindows()
