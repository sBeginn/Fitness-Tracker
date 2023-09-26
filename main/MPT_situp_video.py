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
            percentage_knee = 100
        else:
            line_color_knee = (0, 0, 255)  # Red
            percentage_knee = int((1 - slope_knee) * 100)
            percentage_knee = max(min(percentage_knee, 100), 0)
            if percentage_knee >= 95:
                line_color_knee = (0, 255, 0)  # Green
            elif percentage_knee >= 90:
                line_color_knee = (0, 165, 255)  # Orange
            else:
                line_color_knee = (0, 0, 255)  # Red

        # Draw points and lines on the frame to visualize knee alignment
        cv2.circle(frame, (left_knee_x, left_knee_y), 5, line_color_knee, -1)
        cv2.circle(frame, (right_knee_x, right_knee_y), 5, line_color_knee, -1)
        cv2.line(frame, (left_knee_x, left_knee_y),
                 (right_knee_x, right_knee_y), line_color_knee, 2)

        # Display knee percentage in the top-left corner
        cv2.putText(frame, f'Knee: {percentage_knee}%', (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color_knee, 2)

        return frame, percentage_knee

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

# Main function with rating system
def main(video_path, elbow_threshold_pixels=20, knee_threshold_slope=0.1):
    cap = cv2.VideoCapture(video_path)

    elbow_score = []
    knee_score = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze elbow alignment
        frame, elbow_percentage = analyze_elbow(frame)
        
        # Analyze knee alignment
        frame, knee_percentage = analyze_knees(frame)

        elbow_score.append(elbow_percentage)
        knee_score.append(knee_percentage)

        cv2.imshow('Pose Alignment Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    average_elbow_score = np.mean(elbow_score)
    average_knee_score = np.mean(knee_score)

    elbow_feedback = ""
    if average_elbow_score >= 95:
        elbow_feedback = "Elbow: Very Good!"
    elif average_elbow_score >= 90:
        elbow_feedback = "Elbow: Not exactly straight."
    elif average_elbow_score >= 85:
        elbow_feedback = "Elbow: Not good, please straighten your arms!"
    elif average_elbow_score < 85:
        elbow_feedback = "Elbow: Very bad! Please watch the instruction video!"

    knee_feedback = ""
    if average_knee_score >= 95:
        knee_feedback = "Knees: Very Good!"
    elif average_knee_score >= 90:
        knee_feedback = "Knees: Not exactly straight."
    elif average_knee_score >= 85:
        knee_feedback = "Knees: Not good, please straighten your knees!"
    elif average_knee_score < 85:
        knee_feedback = "Knees: Very bad! Please watch the instruction video!"
        
    overall_score = (average_elbow_score + average_knee_score) / 2

    result = {
        'overall_score': overall_score,
        'elbow_feedback': elbow_feedback,
        'knee_feedback': knee_feedback
    }    
    
    return result



