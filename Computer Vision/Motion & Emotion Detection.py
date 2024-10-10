import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

# Initialize Mediapipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Emotion Detection Function
def detect_emotions(img):
    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        print("DeepFace Analysis Result:", result)  # For debugging
        if isinstance(result, list):
            result = result[0]
        dominant_emotion = result.get('dominant_emotion', None)
        return dominant_emotion
    except Exception as e:
        print("Error detecting emotion:", e)
        return None

# Function to calculate motion based on landmarks
def calculate_motion(landmarks):
    if len(landmarks) > 0:
        coords = np.array([[l.x, l.y] for l in landmarks])
        motion = np.std(coords)  # Calculate movement variance
    else:
        motion = 0
    return motion

# Video Pose Detection Function
def videoDataEx(cap):
    success, img = cap.read()
    if not success:
        print("Failed to capture video frame.")
        return 0, None, None  # Return 0 movement and None for emotion if no video is captured

    # Convert the image to RGB for Mediapipe processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pose Detection
    results = pose.process(img_rgb)
    landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []

    # Calculate motion based on the landmarks
    motion = calculate_motion(landmarks)

    # Draw the landmarks on the image for visualization
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Perform emotion detection on the same frame
    dominant_emotion = detect_emotions(img)

    # Display the emotion and motion on the frame
    cv2.putText(img, f"Emotion: {dominant_emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Motion: {motion:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Dance Floor ppl's Emotion and Motion detector", img)
    cv2.waitKey(1)  # Wait 1ms between frames

    return motion, img, dominant_emotion

# Main function to run video and combine both functionalities
def main():
    cap = cv2.VideoCapture(0)  # Capture video from webcam or replace with video file path

    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    while cap.isOpened():
        movement, img, emotion = videoDataEx(cap)  # Process the frame
        if img is None:  # If no frame is captured, exit the loop
            break

        print(f"Motion: {movement:.2f}, Emotion: {emotion}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
