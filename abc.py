import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from collections import deque

# ----------------------------
# Load model
# ----------------------------
model = tf.keras.models.load_model("model.h5")

# Define your trained class labels
actions = np.array(['hello', 'thanks','yes'])  # 🔧 EDIT to match your dataset

# ----------------------------
# MediaPipe setup
# ----------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Sign Language Detection", layout="wide")
st.title("🧏 Realtime Sign Language Detection")
st.markdown("This app uses your webcam and a trained model to recognize sign language in real time.")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])
prediction_text = st.empty()
sentence_display = st.empty()

# ----------------------------
# Function to extract keypoints
# ----------------------------
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

# ----------------------------
# Load holistic model
# ----------------------------
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sequence = []
predictions = deque(maxlen=10)
sentence = []
threshold = 0.8

# ----------------------------
# Main loop
# ----------------------------
if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Could not access webcam.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract & normalize keypoints
        keypoints = extract_keypoints(results)
        norm = np.linalg.norm(keypoints)
        keypoints = keypoints / norm if norm != 0 else keypoints

        sequence.append(keypoints)
        sequence = sequence[-30:]

        pred_label = ""
        conf = 0.0

        # Prediction block
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(res)
            pred_class = np.argmax(res)
            conf = np.max(res)
            pred_label = actions[pred_class]

            if conf > threshold:
                predictions.append(pred_label)
                # Update sentence smoothly
                if len(predictions) > 2 and predictions[-1] == predictions[-2]:
                    if not sentence or (pred_label != sentence[-1]):
                        sentence.append(pred_label)
                        if len(sentence) > 5:
                            sentence = sentence[-5:]

        # Flip the image for mirror display
        image = cv2.flip(image, 1)

        # Always show prediction text, even if low confidence
        if pred_label != "":
            cv2.putText(image, f'{pred_label} ({conf:.2f})', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update Streamlit UI
        FRAME_WINDOW.image(image)
        if pred_label != "":
            prediction_text.markdown(f"### 🧠 Prediction: **{pred_label}** (Confidence: {conf:.2f})")
        if len(sentence) > 0:
            sentence_display.markdown(f"🗣️ **Recent Signs:** {' '.join(sentence)}")

    cap.release()
else:
    st.info("👆 Click the checkbox above to start the webcam.")
