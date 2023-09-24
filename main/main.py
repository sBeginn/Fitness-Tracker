import streamlit as st
import subprocess
import MPT_pushup_video 
import MPT_squat_video

# Streamlit-Anmeldeseite
st.title('Anmeldung erforderlich')

# Eingabefelder für Benutzername und Passwort
username = st.text_input('Benutzername:')
password = st.text_input('Passwort:', type='password')

# Benutzername und Passwort zum Vergleich
valid_username = "FC"
valid_password = "F"

# Flag, um anzuzeigen, ob der Benutzer angemeldet ist
is_auth = username == valid_username and password == valid_password

if is_auth:
    # Display the username and password as text
    st.write(f'Eingeloggt als Benutzer: {username}')
    st.write('Passwort: ********')  # Display password as asterisks for security

    # Streamlit-Sidebar für die Navigation
    st.sidebar.title('Navigation')
    selected_page = st.sidebar.radio('Wähle eine Seite', ['Dashboard', 'Übungen verwalten'])

    if selected_page == 'Dashboard':
        # Dashboard-Bereich
        st.title('Fitnessübungen')

        # Benutzerfreundliche Meldungen
        st.write("Wähle eine Übung aus und entscheide, ob du sie live mit der Webcam durchführen oder ein Video hochladen möchtest.")

        # Übungen verwalten
        exercises = {
            'Liegestützen': 'Hier kommt die Beschreibung für Liegestützen.',
            'Kniebeugen': 'Hier kommt die Beschreibung für Kniebeugen.',
            'Übung 3': 'Hier kommt die Beschreibung für Übung 3.'
        }
        
        # Create a placeholder for the video analysis results
        result_placeholder = st.empty()

        # Selected exercise and live_or_video option
        selected_exercise = st.selectbox('Übung auswählen', list(exercises.keys()))
        live_or_video = st.radio('Live mit Webcam oder Video', ['Webcam', 'Video'])

        if selected_exercise == 'Liegestützen' and live_or_video == 'Webcam':
            st.write('Du hast "Webcam" ausgewählt. Bitte erlaube den Zugriff auf deine Webcam.')
            if st.button('Starte mit den Liegestützen'):
                subprocess.run(['python', r'C:\Users\Schule\Desktop\MPT_Project\MPT_pushup_webcam.py'])

        if selected_exercise == 'Liegestützen' and live_or_video == 'Video':
            st.write('Du hast "Video" ausgewählt. Lade ein Video hoch:')
            uploaded_file = st.file_uploader("Video-Datei hochladen", type=["mp4", "avi"])
            
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

        if selected_exercise == 'Kniebeugen' and live_or_video == 'Webcam':
            st.write('Du hast "Webcam" ausgewählt. Bitte erlaube den Zugriff auf deine Webcam.')
            if st.button('Starte mit den Kniebeugen'):
                subprocess.run(['python', r'C:\Users\Schule\Desktop\MPT_Project\MPT_squat_webcam.py'])

        if selected_exercise == 'Kniebeugen' and live_or_video == 'Video':
            st.write('Du hast "Video" ausgewählt. Lade ein Video hoch:')
            uploaded_file = st.file_uploader("Video-Datei hochladen", type=["mp4", "avi"])
            
            if uploaded_file is not None:
                # Save the uploaded video to a temporary file
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_file.read())
        
                # Call the main function from the MPT_pushup_video module
                video_path = "temp_video.mp4"  # The path to the uploaded video
                result = MPT_squat_video.main(video_path, shoulder_threshold_meters=0.2, toes_threshold_degrees=15)

                # Display the analysis results
                st.write(f"Overall Performance Score: {result['overall_score']:.2f}%")
                st.write(result['shoulder_feedback'])
                st.write(result['knee_feedback'])
                st.write(result['toes_feedback'])

    elif selected_page == 'Übungen verwalten':
        # Seite für die Verwaltung der Übungen
        st.title('Übungen verwalten')

        # Hinzufügen von Übungen
        new_exercise_name = st.text_input('Name der neuen Übung:')
        new_exercise_description = st.text_area('Beschreibung der neuen Übung:')
        if st.button('Übung hinzufügen'):
            if new_exercise_name and new_exercise_description:
                exercises[new_exercise_name] = new_exercise_description
                st.success(f'Übung "{new_exercise_name}" wurde hinzugefügt.')

        # Liste der vorhandenen Übungen anzeigen
        st.write('**Liste der vorhandenen Übungen:**')
        st.write(exercises)
