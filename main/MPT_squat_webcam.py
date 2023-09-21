import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate pixel coordinates 
def calculate_landmark_pixel(landmark, frame_shape):
    x, y = int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0])
    return x, y

# Function for analyze shoulder 
def analyze_shoulder(frame, shoulder_threshold_pixels):
    height, width, _ = frame.shape

    # Process frame using Pose estimation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks

    if landmarks:
        # This are left and right shoulder landmarks
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate pixel coordinates for shoulders
        left_shoulder_x, left_shoulder_y = calculate_landmark_pixel(left_shoulder, frame.shape)
        right_shoulder_x, right_shoulder_y = calculate_landmark_pixel(right_shoulder, frame.shape)

        # Calculate slope of  the shoulders
        slope = abs((right_shoulder_y - left_shoulder_y) / (right_shoulder_x - left_shoulder_x + 1e-5))
        
        # Calculate color and percentage of shoulders
        if slope <= 0.1 and abs(left_shoulder_x - right_shoulder_x) <= shoulder_threshold_pixels:
            line_color = (0, 255, 0)  # Green
            percentage = 100
        else:
            line_color = (0, 0, 255)  # Red
            percentage = int((1 - slope) * 100)
            percentage = max(min(percentage, 100), 0)
            if percentage >= 95:
                line_color = (0, 255, 0)  # Green
            elif percentage >= 90:
                line_color = (0, 165, 255)  # Orange
            else:
                line_color = (0, 0, 255)  # Red

        cv2.circle(frame, (left_shoulder_x, left_shoulder_y), 5, line_color, -1)
        cv2.circle(frame, (right_shoulder_x, right_shoulder_y), 5, line_color, -1)

        cv2.line(frame, (left_shoulder_x, left_shoulder_y),
                 (right_shoulder_x, right_shoulder_y), line_color, 2)

        cv2.putText(frame, f'Shoulder: {percentage}%', (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)

    return frame

# Function for analyze knee 
def analyze_knee(frame):
    height, width, _ = frame.shape

    # Process frame using Pose estimation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks

    if landmarks:
        # This are left and right knee landmarks
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

        # Calculate pixel coordinates for knees
        left_knee_x, left_knee_y = calculate_landmark_pixel(left_knee, frame.shape)
        right_knee_x, right_knee_y = calculate_landmark_pixel(right_knee, frame.shape)

        # Calculate slope of the  knees
        slope_knee = abs((right_knee_y - left_knee_y) / (right_knee_x - left_knee_x + 1e-5))

        # # Calculate color and percentage of knee
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

        cv2.circle(frame, (left_knee_x, left_knee_y), 5, line_color_knee, -1)
        cv2.circle(frame, (right_knee_x, right_knee_y), 5, line_color_knee, -1)

        cv2.line(frame, (left_knee_x, left_knee_y),
                 (right_knee_x, right_knee_y), line_color_knee, 2)

        cv2.putText(frame, f'Knee: {percentage_knee}%', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color_knee, 2)

    return frame

# Function for analyze toes 
def analyze_toes(frame, toes_threshold_degrees=1):
    height, width, _ = frame.shape

    # Process frame using Pose estimation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks

    if landmarks:
        # This are left and right toes landmarks (Left & Right swapped because of mirror-inverted)
        left_toes = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        right_toes = landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

        # Calculate pixel coordinates for toes
        left_toes_x, left_toes_y = calculate_landmark_pixel(left_toes, frame.shape)
        right_toes_x, right_toes_y = calculate_landmark_pixel(right_toes, frame.shape)

        # Calculate the angle between the toes and horizontal axis
        angle_radians = np.arctan2(right_toes_y - left_toes_y, right_toes_x - left_toes_x)
        angle_degrees = np.degrees(angle_radians)

        # # Calculate color and percentage of toe
        if abs(angle_degrees) <= toes_threshold_degrees:
            line_color_toes = (0, 255, 0)  # Green
            percentage_toes = 100
        else:
            line_color_toes = (0, 0, 255)  # Red
            percentage_toes = int(100 - abs(angle_degrees))
            percentage_toes = max(min(percentage_toes, 100), 0)
            if percentage_toes >= 95:
                line_color_toes = (0, 255, 0)  # Green
            elif percentage_toes >= 90:
                line_color_toes = (0, 165, 255)  # Orange
            else:
                line_color_toes = (0, 0, 255)  # Red

        cv2.circle(frame, (left_toes_x, left_toes_y), 5, line_color_toes, -1)
        cv2.circle(frame, (right_toes_x, right_toes_y), 5, line_color_toes, -1)
        
        cv2.line(frame, (left_toes_x, left_toes_y),
        (right_toes_x, right_toes_y), line_color_toes, 2)

        # For position of text
        text_x = int((width - 150) / 2)  # Calculate the center position
        text_y = 30
        
        cv2.putText(frame, f'Toes: {percentage_toes}%', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color_toes, 2)

    return frame


# Open Webcam
cap = cv2.VideoCapture(0)


# Settings for 1-meter distance to the camera
shoulder_width_threshold_meters = 0.2  
focal_length = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
shoulder_width_threshold_pixels = (shoulder_width_threshold_meters * focal_length) / 1  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally
    flipped_frame = cv2.flip(frame, 1)

    analyzed_frame = analyze_shoulder(flipped_frame, shoulder_width_threshold_pixels)
    
    analyzed_frame = analyze_knee(analyzed_frame)

    analyzed_frame = analyze_toes(analyzed_frame)


    cv2.imshow('Squat Form Analysis', analyzed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
