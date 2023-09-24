import cv2
import mediapipe as mp
import numpy as np

# Initialize the MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to analyze knee alignment
def analyze_knees(frame):
    # Convert the frame to RGB format for pose estimation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using the Pose estimation model
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks

    # Initialize knee_percentage with a default value
    knee_percentage = 0

    if landmarks:
        # Extract left and right knee landmarks
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        height, width, _ = frame.shape

        # Calculate pixel coordinates for knees
        left_knee_x = int(left_knee.x * width)
        left_knee_y = int(left_knee.y * height)
        right_knee_x = int(right_knee.x * width)
        right_knee_y = int(right_knee.y * height)

        # Calculate the slope of the line connecting the knees
        slope_knee = abs((right_knee_y - left_knee_y) / (right_knee_x - left_knee_x + 1e-5))

        # Determine color and percentage based on knee alignment
        if slope_knee <= 0.1:
            line_color_knee = (0, 255, 0)  # Green
            knee_percentage = 100
        else:
            line_color_knee = (0, 0, 255)  # Red
            knee_percentage = int((1 - slope_knee) * 100)
            knee_percentage = max(min(knee_percentage, 100), 0)
            if knee_percentage >= 95:
                line_color_knee = (0, 255, 0)  # Green
            elif knee_percentage >= 90:
                line_color_knee = (0, 165, 255)  # Orange
            else:
                line_color_knee = (0, 0, 255)  # Red

        # Draw points and lines on the frame to visualize knee alignment
        cv2.circle(frame, (left_knee_x, left_knee_y), 5, line_color_knee, -1)
        cv2.circle(frame, (right_knee_x, right_knee_y), 5, line_color_knee, -1)
        cv2.line(frame, (left_knee_x, left_knee_y),
                 (right_knee_x, right_knee_y), line_color_knee, 2)

        # Display knee percentage in the top-left corner
        cv2.putText(frame, f'Knee: {knee_percentage}%', (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color_knee, 2)

    # Always return the frame and the value of knee_percentage
    return frame, knee_percentage

# Function to analyze elbow alignment
def analyze_elbow(frame):
    # Convert the frame to RGB format for pose estimation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using the Pose estimation model
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks

    # Initialize elbow_percentage with a default value
    elbow_percentage = 0

    if landmarks:
        # Extract left and right elbow landmarks
        left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        height, width, _ = frame.shape

        # Calculate pixel coordinates for elbows
        left_elbow_x = int(left_elbow.x * width)
        left_elbow_y = int(left_elbow.y * height)
        right_elbow_x = int(right_elbow.x * width)
        right_elbow_y = int(right_elbow.y * height)

        # Calculate the slope of the line connecting the elbows
        slope_elbow = abs((right_elbow_y - left_elbow_y) / (right_elbow_x - left_elbow_x + 1e-5))

        # Determine color and percentage based on elbow alignment
        if slope_elbow <= 0.1:
            line_color_elbow = (0, 255, 0)  # Green
            elbow_percentage = 100
        else:
            line_color_elbow = (0, 0, 255)  # Red
            elbow_percentage = int((1 - slope_elbow) * 100)
            elbow_percentage = max(min(elbow_percentage, 100), 0)
            if elbow_percentage >= 95:
                line_color_elbow = (0, 255, 0)  # Green
            elif elbow_percentage >= 90:
                line_color_elbow = (0, 165, 255)  # Orange
            else:
                line_color_elbow = (0, 0, 255)  # Red

        # Draw points and lines on the frame to visualize elbow alignment
        cv2.circle(frame, (left_elbow_x, left_elbow_y), 5, line_color_elbow, -1)
        cv2.circle(frame, (right_elbow_x, right_elbow_y), 5, line_color_elbow, -1)
        cv2.line(frame, (left_elbow_x, left_elbow_y),
                 (right_elbow_x, right_elbow_y), line_color_elbow, 2)

        # Display elbow percentage in the top-left corner
        cv2.putText(frame, f'Elbow: {elbow_percentage}%', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color_elbow, 2)

    # Always return the frame and the value of elbow_percentage
    return frame, elbow_percentage

# Main function
def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)

    while cap.isOpened():
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)

        # Analyze knee alignment
        flipped_frame, knee_percentage = analyze_knees(flipped_frame)

        # Analyze elbow alignment
        flipped_frame, elbow_percentage = analyze_elbow(flipped_frame)

        # Display the frame with feedback
        cv2.imshow('Pose Alignment Analysis', flipped_frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Call the main function
    main()
