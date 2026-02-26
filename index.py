import cv2
import mediapipe as mp
import sounddevice as sd
import soundfile as sf

# The playlist
playlist = [
    {"video": "media/triggering_breakup.mov", "audio": "media/triggering_breakup.wav"},
    {"video": "media/pinky_up.mov", "audio": "media/pinky_up.wav"},
    {"video": "media/squat.mov", "audio": "media/squat.wav"}
]

current_idx = 0
audio_data, fs = None, None

def load_media(idx):
    global audio_data, fs
    item = playlist[idx]
    audio_data, fs = sf.read(item["audio"], dtype='float32')
    return cv2.VideoCapture(item["video"])

# Initial Setup
video_cap = load_media(current_idx)
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_window_name = "GET BACK TO WORK"
audio_playing = False
was_looking_away = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    looking_away = True 
    if results.multi_face_landmarks:
        nose = results.multi_face_landmarks[0].landmark[4]
        if 0.35 < nose.x < 0.65 and 0.35 < nose.y < 0.65:
            looking_away = False

    # Detect the moment you look away
    if looking_away and not was_looking_away:
        video_cap.release()
        video_cap = load_media(current_idx)
        
        # Start Audio
        sd.play(audio_data, fs, loop=True)
        audio_playing = True
        
        # Move to next video
        current_idx = (current_idx + 1) % len(playlist)

    if looking_away:
        ret, v_frame = video_cap.read()
        # If video ends naturally while looking away, loop THIS specific video
        if not ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, v_frame = video_cap.read()

        if ret:
            v_frame = cv2.resize(v_frame, (360, 640)) 
            cv2.imshow(video_window_name, v_frame)
    else:
        # Stopped looking away
        if audio_playing:
            sd.stop()
            audio_playing = False
            try: cv2.destroyWindow(video_window_name)
            except: pass

    # Update state for next frame comparison
    was_looking_away = looking_away

    # Status UI
    status_color = (0, 0, 255) if looking_away else (0, 255, 0)
    cv2.putText(frame, f"STATUS: {'DISTRACTED' if looking_away else 'FOCUSED'}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.imshow("Webcam Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
video_cap.release()
sd.stop()
cv2.destroyAllWindows()