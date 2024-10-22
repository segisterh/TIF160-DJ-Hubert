import cv2
import torch
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # Filter for person class only

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_motion(landmarks_prev, landmarks_curr, threshold=0.05):
    if landmarks_prev is not None and landmarks_curr is not None:
        prev_coords = np.array([[l.x, l.y] for l in landmarks_prev.landmark])
        curr_coords = np.array([[l.x, l.y] for l in landmarks_curr.landmark])
        motion = np.linalg.norm(curr_coords - prev_coords)
        return 1 if motion >= threshold else 0
    return 0

def detect_emotions(img, face_locations):
    emotions = []
    try:
        for (top, right, bottom, left) in face_locations:
            face_img = img[top:bottom, left:right]
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            dominant_emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
            emotions.append(1 if dominant_emotion in ['happy', 'surprise'] else 0)
    except Exception as e:
        print("Error detecting emotion:", e)
    return emotions

def process_frame(frame, prev_landmarks_list):
    # YOLOv5 person detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    
    current_landmarks_list = []
    motion_output = []
    face_locations = []

    for detection in detections:
        if detection[5] == 0:  # Class 0 is person
            x1, y1, x2, y2 = map(int, detection[:4])
            person_frame = frame[y1:y2, x1:x2]
            face_locations.append((y1, x2, y2, x1))  # top, right, bottom, left
            
            # MediaPipe pose estimation
            results_pose = pose.process(cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB))
            
            if results_pose.pose_landmarks:
                current_landmarks_list.append(results_pose.pose_landmarks)
                
                # Calculate motion
                prev_landmarks = prev_landmarks_list[len(current_landmarks_list)-1] if len(current_landmarks_list) <= len(prev_landmarks_list) else None
                motion = calculate_motion(prev_landmarks, results_pose.pose_landmarks)
                
                motion_output.append(motion)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                motion_output.append(0)  # No motion detected

    # Emotion detection
    emotions = detect_emotions(frame, face_locations)
    
    # Fill in emotion output for people without detected faces
    emotion_output = [1 if i >= len(emotions) else emotions[i] for i in range(len(motion_output))]

    return frame, current_landmarks_list, motion_output, emotion_output

def main():
    cap = cv2.VideoCapture(0)  # Use 0 if this is your primary camera
    prev_landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame, current_landmarks_list, motion_output, emotion_output = process_frame(frame, prev_landmarks_list)

        cv2.imshow('Multi-Person Detection', frame)

        print("Motion Output:", motion_output)
        print("Emotion Output:", emotion_output)

        prev_landmarks_list = current_landmarks_list

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()