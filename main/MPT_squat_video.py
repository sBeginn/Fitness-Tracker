import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function for analyze toe 
def analyze_toes(frame):
    # Convert to RGB format 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using the Pose estimation model
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks

    if landmarks:
        left_toe = landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_toe = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        height, width, _ = frame.shape

        left_toe_x = int(left_toe.x * width)
        left_toe_y = int(left_toe.y * height)
        right_toe_x = int(right_toe.x * width)
        right_toe_y = int(right_toe.y * height)


        slope_toe = abs((right_toe_y - left_toe_y) / (right_toe_x - left_toe_x + 1e-5))


        if slope_toe <= 0.1:
            line_color_toe = (0, 255, 0)  # Green
            percentage_toe = 100
        else:
            line_color_toe = (0, 0, 255)  # Red
            percentage_toe = int((1 - slope_toe) * 100)
            percentage_toe = max(min(percentage_toe, 100), 0)
            if percentage_toe >= 95:
                line_color_toe = (0, 255, 0)  # Green
            elif percentage_toe >= 90:
                line_color_toe = (0, 165, 255)  # Orange
            else:
                line_color_toe = (0, 0, 255)  # Red


        cv2.circle(frame, (left_toe_x, left_toe_y), 5, line_color_toe, -1)
        cv2.circle(frame, (right_toe_x, right_toe_y), 5, line_color_toe, -1)
        cv2.line(frame, (left_toe_x, left_toe_y),
                 (right_toe_x, right_toe_y), line_color_toe, 2)

        text_width_toes, _ = cv2.getTextSize(f'Toes: {percentage_toe}%', cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x_coord_toes = (width - text_width_toes[0]) // 2

        cv2.putText(frame, f'Toes: {percentage_toe}%', (x_coord_toes, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color_toe, 2)  # Centered on top

        return frame, percentage_toe



# Function for analyze shoulder 
def analyze_shoulders(frame, shoulder_threshold_pixels):
    # Convert to RGB format 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using the Pose estimation model
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks

    if landmarks:

        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        height, width, _ = frame.shape


        left_shoulder_x = int(left_shoulder.x * width)
        left_shoulder_y = int(left_shoulder.y * height)
        right_shoulder_x = int(right_shoulder.x * width)
        right_shoulder_y = int(right_shoulder.y * height)

        slope = abs((right_shoulder_y - left_shoulder_y) / (right_shoulder_x - left_shoulder_x + 1e-5))


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

        return frame, percentage

# Function for analyze knee
def analyze_knees(frame):
    # Convert to RGB format 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using the Pose estimation model
    results = pose.process(rgb_frame)
    landmarks = results.pose_landmarks

    if landmarks:
        # This are left and right knee landmarks
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        height, width, _ = frame.shape

        # Coordinates for knees
        left_knee_x = int(left_knee.x * width)
        left_knee_y = int(left_knee.y * height)
        right_knee_x = int(right_knee.x * width)
        right_knee_y = int(right_knee.y * height)

        slope_knee = abs((right_knee_y - left_knee_y) / (right_knee_x - left_knee_x + 1e-5))

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

        return frame, percentage_knee

# Main function
def main(video_path, shoulder_threshold_meters=0.2, toes_threshold_degrees=15):
    cap = cv2.VideoCapture(video_path)

    focal_length = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    shoulder_width_threshold_pixels = (shoulder_threshold_meters * focal_length) / 1

    shoulder_score = []
    knee_score = []
    toes_score = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally
        flipped_frame = cv2.flip(frame, 1)


        flipped_frame, shoulder_percentage = analyze_shoulders(flipped_frame, shoulder_width_threshold_pixels)

        flipped_frame, knee_percentage = analyze_knees(flipped_frame)
        
        flipped_frame, toes_percentage = analyze_toes(flipped_frame)
        

        shoulder_score.append(shoulder_percentage)
        knee_score.append(knee_percentage)
        toes_score.append(toes_percentage)

        cv2.imshow('Squat Form Analysis', flipped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    average_shoulder_score = np.mean(shoulder_score)
    average_knee_score = np.mean(knee_score)
    average_toes_score = np.mean(toes_score)

    shoulder_feedback = ""
    if average_shoulder_score >= 95:
        shoulder_feedback = "Shoulders: Very Good!"
    elif average_shoulder_score >= 90:
        shoulder_feedback = "Shoulders: Not exactly straight."
    elif average_shoulder_score >= 85:
        shoulder_feedback = "Shoulders: Not good, please straighter!"

    knee_feedback = ""
    if average_knee_score >= 95:
        knee_feedback = "Knees: Very Good!"
    elif average_knee_score >= 90:
        knee_feedback = "Knees: Not exactly straight."
    elif average_knee_score >= 85:
        knee_feedback = "Knees: Not good, please straighter!"
        
    toes_feedback = ""
    if average_toes_score >= 95:
        toes_feedback = "Toes: Very Good!"
    elif average_toes_score >= 90:
        toes_feedback = "Toes: Not exactly straight."
    elif average_toes_score >= 85:
        toes_feedback = "Toes: Not good, please straighten your feet!"
        
    overall_score = (average_shoulder_score + average_knee_score + average_toes_score) / 3

    print(f"Overall Performance Score: {overall_score:.2f}%")
    print(shoulder_feedback)
    print(knee_feedback)
    print(toes_feedback)

if __name__ == "__main__":
    video_path = r"C:\Users\Schule\Desktop\MPT_Project\WhatsApp Video 2023-08-30 at 11.58.27.mp4"
    main(video_path)