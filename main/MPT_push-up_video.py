import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function for analyze hands
def analyze_hands(frame, hand_distance_threshold_pixels):
    
    # Extract hieght and width
    height, width, _ = frame.shape
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks
    
    if landmarks:
        #Defined landmarks for hands
        left_fingertip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        right_fingertip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        
        left_fingertip_x = int(left_fingertip.x * width)
        left_fingertip_y = int(left_fingertip.y * height)
        right_fingertip_x = int(right_fingertip.x * width)
        right_fingertip_y = int(right_fingertip.y * height)
        
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
    
    return frame, percentage

# Function for analyze shoulders
def analyze_shoulder(frame, shoulder_threshold_pixels):
    
    # Extract height and width
    height, width, _ = frame.shape
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks
    
    if landmarks:
        # Defined landmarks for shoulders
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        left_shoulder_x = int(left_shoulder.x * width)
        left_shoulder_y = int(left_shoulder.y * height)
        right_shoulder_x = int(right_shoulder.x * width)
        right_shoulder_y = int(right_shoulder.y * height)
        
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
    
    return frame, percentage

# Main function
def main(video_path, shoulder_threshold_meters=0.2, fingertips_threshold_pixels=50):
    cap = cv2.VideoCapture(video_path)
    
    focal_length = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    shoulder_width_threshold_pixels = (shoulder_threshold_meters * focal_length) / 1
    
    #Scores
    shoulder_score = []
    hand_score = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        flipped_frame = cv2.flip(frame, 1)
        width = flipped_frame.shape[1]
        
        flipped_frame, shoulder_percentage = analyze_shoulder(flipped_frame, shoulder_width_threshold_pixels)
        flipped_frame, hand_percentage = analyze_hands(flipped_frame, fingertips_threshold_pixels)
        
        shoulder_score.append(shoulder_percentage)
        hand_score.append(hand_percentage)
        
        cv2.imshow('Alignment Analysis', flipped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    #Average Scores
    average_shoulder_score = np.mean(shoulder_score)
    average_hand_score = np.mean(hand_score)
    
    #Feedback
    shoulder_feedback = ""
    if average_shoulder_score >= 95:
        shoulder_feedback = "Shoulders: Very Good!"
    elif average_shoulder_score >= 90:
        shoulder_feedback = "Shoulders: Not exactly straight."
    elif average_shoulder_score >= 85:
        shoulder_feedback = "Shoulders: Not good, please straighten your shoulders!"

    hand_feedback = ""
    if average_hand_score >= 95:
        hand_feedback = "Hands: Very Good!"
    elif average_hand_score >= 90:
        hand_feedback = "Hands: Not exactly aligned."
    elif average_hand_score >= 85:
        hand_feedback = "Hands: Not good, please align your hands!"

    overall_score = (average_shoulder_score + average_hand_score) / 2
    
    print(f"Overall Performance Score: {overall_score:.2f}%")
    print(shoulder_feedback)
    print(hand_feedback)

if __name__ == "__main__":
    video_path = r"C:\Users\Schule\Desktop\MPT_Project\WhatsApp Video 2023-08-30 at 12.03.49.mp4"
    main(video_path, shoulder_threshold_meters=0.2, fingertips_threshold_pixels=50)
