import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function for pixel image
def calculate_landmark_pixel(landmark, frame_shape):
    x, y = int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0])
    return x, y

# Function for analyze shoulder
def analyze_shoulder(frame, shoulder_threshold_pixels):
    # Extract height and width
    height, width, _ = frame.shape
    
    # Convert th RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks
    
    if landmarks:
        #Defined landmarks for shoulders
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        left_shoulder_x, left_shoulder_y = calculate_landmark_pixel(left_shoulder, frame.shape)
        right_shoulder_x, right_shoulder_y = calculate_landmark_pixel(right_shoulder, frame.shape)
        
        slope = abs((right_shoulder_y - left_shoulder_y) / (right_shoulder_x - left_shoulder_x + 1e-5))
        
        if slope <= 0.1 and abs(left_shoulder_x - right_shoulder_x) <= shoulder_threshold_pixels:
            line_color = (0, 255, 0)
            percentage = 100
        else:
            line_color = (0, 0, 255)
            percentage = int((1 - slope) * 100)
            percentage = max(min(percentage, 100), 0)
            if percentage >= 95:
                line_color = (0, 255, 0)
            elif percentage >= 90:
                line_color = (0, 165, 255)
            else:
                line_color = (0, 0, 255)
                
        cv2.circle(frame, (left_shoulder_x, left_shoulder_y), 5, line_color, -1)
        cv2.circle(frame, (right_shoulder_x, right_shoulder_y), 5, line_color, -1)
        
        cv2.line(frame, (left_shoulder_x, left_shoulder_y), 
                 (right_shoulder_x, right_shoulder_y), line_color, 2)
        
        cv2.putText(frame, f'Shoulder: {percentage}%', (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
    
    return frame

# Function for analyze hands
def analyze_hands(frame, hand_distance_threshold_pixels):
    # Extract height and width
    height, width, _ = frame.shape
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks
    
    if landmarks:
        # Landmarks swapped because of mirror-inverted
        left_fingertip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        right_fingertip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        
        left_fingertip_x, left_fingertip_y = calculate_landmark_pixel(left_fingertip, frame.shape)
        right_fingertip_x, right_fingertip_y = calculate_landmark_pixel(right_fingertip, frame.shape)
        
        angle_degrees = np.degrees(np.arctan2(right_fingertip_y - left_fingertip_y, right_fingertip_x - left_fingertip_x))
        
        if -15 <= angle_degrees <= 15 and abs(left_fingertip_y - right_fingertip_y) <= hand_distance_threshold_pixels:
            line_color = (0, 255, 0)
            percentage = 100
        else:
            line_color = (0, 0, 255)
            percentage = int((1 - abs(angle_degrees) / 15) * 100)
       
        cv2.circle(frame, (left_fingertip_x, left_fingertip_y), 5, line_color, -1)
        cv2.circle(frame, (right_fingertip_x, right_fingertip_y), 5, line_color, -1)
       
        cv2.line(frame, (left_fingertip_x, left_fingertip_y), 
                 (right_fingertip_x, right_fingertip_y), line_color, 2)
        
        cv2.putText(frame, f'Hand: {percentage}%', (width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
    return frame

#Distance for cam (Now 1 meter )
shoulder_width_threshold_meters = 0.2
hand_distance_threshold_pixels = 5

cap = cv2.VideoCapture(0)

focal_length = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

shoulder_width_threshold_pixels = (shoulder_width_threshold_meters * focal_length) / 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    flipped_frame = cv2.flip(frame, 1)
    
    analyzed_frame = analyze_shoulder(flipped_frame, shoulder_width_threshold_pixels)
    analyzed_frame = analyze_hands(analyzed_frame, hand_distance_threshold_pixels)
    
    cv2.imshow('Alignment Analysis', analyzed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

