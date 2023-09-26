import streamlit as st
import subprocess
import MPT_pushup_video
import MPT_squat_video
import MPT_situp_video  

#Streamlit app title
st.title('Fitness Tracker')

# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True)

#Dashboard section
st.title('Choose your Exercises')


#Exercise information and tutorial
exercise_info = {
    'Push-ups': {
        'description': 'Push-ups are a great upper body exercise that target the chest, shoulders, and triceps.',
        'video_path': r"C:\Users\Schule\Desktop\MPT_Project\pushup.mp4" ,  #Local File
        'tutorial': [
            '1. Start in a high plank position with your hands slightly wider than shoulder-width apart.',
            '2. Lower your body until your chest touches the ground, keeping your back straight.',
            '3. Push your body back up to the starting position, fully extending your arms.'
        ],
        'instruction_video_path': r"C:\Users\Schule\Desktop\MPT_Project\pushup.mp4"  
    },
    'Sit-ups': {
        'description': 'Sit-ups are a core strengthening exercise that target the abdominal muscles.',
        'video_path': r"C:\Users\Schule\Desktop\MPT_Project\situp.mp4",  #Local File
        'tutorial': [
            '1. Lie on your back with your knees bent and feet flat on the ground.',
            '2. Place your hands behind your head or cross your arms over your chest.',
            '3. Engage your core muscles and lift your upper body off the ground, bringing your chest toward your knees.',
            '4. Lower your upper body back to the ground.'
        ],
        'instruction_video_path': r"C:\Users\Schule\Desktop\MPT_Project\situp.mp4" 
    },
    'Squats': {
        'description': 'Squats are a lower body exercise that target the quadriceps, hamstrings, and glutes.',
        'video_path': r"C:\Users\Schule\Desktop\MPT_Project\squat.mp4",  #Local File
        'tutorial': [
            '1. Stand with your feet shoulder-width apart and toes slightly turned out.',
            '2. Keep your chest up and back straight as you lower your body by bending your knees and hips.',
            '3. Go as low as you can while maintaining good form, ideally until your thighs are parallel to the ground.',
            '4. Push through your heels to return to the starting position.'
        ],
        'instruction_video_path': r"C:\Users\Schule\Desktop\MPT_Project\squat.mp4"
    }
}


# Create a placeholder for the video analysis results
result_placeholder = st.empty()

# Selected exercise and live_or_video option
selected_exercise = st.selectbox('Select an Exercise', list(exercise_info.keys()))

# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True)

# Display exercise description and tutorial
if selected_exercise in exercise_info:
    exercise_details = exercise_info[selected_exercise]
    st.subheader(selected_exercise)
    st.write(exercise_details['description'])
    st.subheader('Exercise Tutorial:')
    for step in exercise_details['tutorial']:
        st.write(step)
  
    # Add a horizontal rule using HTML
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display instruction video if available
    if 'instruction_video_path' in exercise_details:
        st.subheader('Instruction Video:')
        st.video(exercise_details['instruction_video_path'])

# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True)

live_or_video = st.radio('Perform Live with Webcam or Upload a Video', ['Webcam', 'Video'])

if selected_exercise == 'Push-ups' and live_or_video == 'Webcam':
   st.write('You have selected "Webcam." Please follow these instructions:')
   st.write('1. Make sure your webcam is properly set up and functioning.')
   st.write('2. Position yourself about 1-2 meters away from the webcam so that your whole body is visible.')
   st.write('3. Ensure there is enough lighting in the room for clear visibility.')
   st.write('4. Click the "Start" button when you are ready to begin the exercise.')
   st.write('5. Your browser may ask for permission to access your camera. Please allow access for the webcam to work properly.')
   if st.button('Start Push-ups'):
        subprocess.run(['python', r"C:\Users\Schule\Desktop\MPT_Project\MPT_pushup_webcam.py"])

if selected_exercise == 'Push-ups' and live_or_video == 'Video':
    st.write('You have selected "Video." Upload a video:')
    uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
    
        # Call the main function from the MPT_pushup_video module
        video_path = "temp_video.mp4"  # The path to the uploaded video
        result = MPT_pushup_video.main(video_path, shoulder_threshold_meters=0.2, fingertips_threshold_pixels=50)

        # Display the analysis results
        st.write(f"Overall Performance Score: {result['overall_score']:.2f}%")
        st.write(result['shoulder_feedback'])
        st.write(result['hand_feedback'])

if selected_exercise == 'Squats' and live_or_video == 'Webcam':
   st.write('You have selected "Webcam." Please follow these instructions:')
   st.write('1. Make sure your webcam is properly set up and functioning.')
   st.write('2. Position yourself about 1-2 meters away from the webcam so that your whole body is visible.')
   st.write('3. Ensure there is enough lighting in the room for clear visibility.')
   st.write('4. Click the "Start" button when you are ready to begin the exercise.')
   st.write('5. Your browser may ask for permission to access your camera. Please allow access for the webcam to work properly.')
   if st.button('Start Squats'):
        subprocess.run(['python', r"C:\Users\Schule\Desktop\MPT_Project\MPT_squat_webcam.py"])

if selected_exercise == 'Squats' and live_or_video == 'Video':
    st.write('You have selected "Video." Upload a video:')
    uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
    
        # Call the main function from the MPT_squat_video module
        video_path = "temp_video.mp4"  # The path to the uploaded video
        result = MPT_squat_video.main(video_path, shoulder_threshold_meters=0.2, toes_threshold_degrees=15)

        # Display the analysis results
        st.write(f"Overall Performance Score: {result['overall_score']:.2f}%")
        st.write(result['shoulder_feedback'])
        st.write(result['knee_feedback'])
        st.write(result['toes_feedback'])

# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True)


if selected_exercise == 'Sit-ups' and live_or_video == 'Webcam':
   st.write('You have selected "Webcam." Please follow these instructions:')
   st.write('1. Make sure your webcam is properly set up and functioning.')
   st.write('2. Position yourself about 1-2 meters away from the webcam so that your whole body is visible.')
   st.write('3. Ensure there is enough lighting in the room for clear visibility.')
   st.write('4. Click the "Start" button when you are ready to begin the exercise.')
   st.write('5. Your browser may ask for permission to access your camera. Please allow access for the webcam to work properly.')
   if st.button('Start Sit-ups'):
        subprocess.run(['python', r"C:\Users\Schule\Desktop\MPT_Project\MPT_SitUp_webcam.py"])

if selected_exercise == 'Sit-ups' and live_or_video == 'Video':
    st.write('You have selected "Video." Upload a video:')
    uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
    
        # Call the main function from the MPT_situp_video module
        video_path = "temp_video.mp4"  # The path to the uploaded video
        result = MPT_situp_video.main(video_path)  # You can pass any required parameters here

        # Display the analysis results
        st.write(f"Overall Performance Score: {result['overall_score']:.2f}%")
        st.write(result['knee_feedback'])
        st.write(result['elbow_feedback'])
        # Add other feedback or results specific to Sit-ups
