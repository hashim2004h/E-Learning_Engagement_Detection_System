import re
import json
import random
import string
import secrets
import smtplib
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO   # <-- we’re also using this below for ReportLab
from collections import deque

from bson import ObjectId, SON
import numpy as np
import cv2
import mediapipe as mp
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, Response, send_file  # <-- added send_file
)
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from flask_socketio import SocketIO, emit, join_room
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from email.message import EmailMessage
from PIL import Image

# ↓↓↓ newly added ReportLab imports:
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO
from flask import send_file

# (Any “WeasyPrint” import should be removed/commented out.)



app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = 'your-secret-key-here'  # Change for production

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# MongoDB setup
client = MongoClient(config['MONGO_URI'])
db = client[config['DB_NAME']]
users = db.users
otps = db.otps
sessions = db.sessions
meetings = db.meetings

# Global engagement log (for the session)
engagement_log = []

# ---------------------------
# MediaPipe Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Eye landmark indices and iris indices.
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
IRIS_IDX = {
    'left': [474, 475, 476, 477],
    'right': [469, 470, 471, 472]
}

# 3D Head pose model points.
HEAD_POSE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -0.1, 0.0),       # Chin
    (-0.05, 0.03, 0.0),     # Left eye left corner
    (0.05, 0.03, 0.0),      # Right eye right corner
    (-0.03, -0.05, 0.0),    # Left mouth corner
    (0.03, -0.05, 0.0)      # Right mouth corner
], dtype=np.float64)

# Thresholds & State Variables.
EAR_THRESHOLD = 0.22
EAR_HYSTERESIS = 0.05
GAZE_THRESHOLD = 20
HEAD_POSE_THRESHOLD = 25
GAZE_HORIZONTAL_NORM_THRESHOLD = 0.25
BLINK_FRAME_THRESHOLD = 5

eyes_closed_frames = 0
blink_count = 0
ear_history = deque(maxlen=10)
yaw_history = deque(maxlen=10)
pitch_history = deque(maxlen=10)
last_eye_state = True

prev_time = time.time()
fps = 0

video_streaming_active = True

# Global metrics dictionary.
current_metrics = {
    'engagement': 'N/A',
    'blinks': 'N/A',
    'fps': 'N/A',
    'pitch': 'N/A',
    'yaw': 'N/A',
    'roll': 'N/A',
    'ear': 'N/A'
}

# Initialize camera.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
@socketio.on('update_noise')
def handle_update_noise(data):
    noise = data.get('noise', 'N/A')
    current_metrics['noise_level'] = noise
    print("Updated noise_level:", noise)  # For debugging

@socketio.on('update_clicks')
def handle_update_clicks(data):
    clicks = data.get('clicks', 'N/A')
    current_metrics['mouse_clicks'] = clicks
    print("Updated mouse_clicks:", clicks)  # For debugging

# Utility functions.
def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def compute_ear(landmarks, eye_indices, image_size):
    w, h = image_size
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    vert1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    vert2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    horiz = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    if horiz < 1e-6:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)

def compute_3d_gaze(landmarks, eye_side, image_size):
    h, w = image_size
    eye_indices = LEFT_EYE_IDX if eye_side == 'left' else RIGHT_EYE_IDX
    iris_indices = IRIS_IDX[eye_side]
    eye_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices])
    iris_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices])
    eye_center = np.mean(eye_points, axis=0)
    iris_center = np.mean(iris_points, axis=0)
    gaze_vector = iris_center - eye_center
    return gaze_vector, eye_center, iris_center

def compute_normalized_horizontal_gaze(landmarks, eye_side, image_size):
    h, w = image_size
    eye_indices = LEFT_EYE_IDX if eye_side == 'left' else RIGHT_EYE_IDX
    iris_indices = IRIS_IDX[eye_side]
    eye_x = [landmarks[i].x * w for i in eye_indices]
    iris_x = [landmarks[i].x * w for i in iris_indices]
    eye_center = np.mean(eye_x)
    eye_width = max(eye_x) - min(eye_x)
    if eye_width < 1e-6:
        return 0.0
    iris_center = np.mean(iris_x)
    return (iris_center - eye_center) / eye_width

def compute_head_pose(landmarks, frame):
    h, w = frame.shape[:2]
    image_points = np.array([
        (landmarks[4].x * w, landmarks[4].y * h),      # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h),    # Chin
        (landmarks[33].x * w, landmarks[33].y * h),      # Left eye left corner
        (landmarks[263].x * w, landmarks[263].y * h),    # Right eye right corner
        (landmarks[61].x * w, landmarks[61].y * h),      # Left mouth corner
        (landmarks[291].x * w, landmarks[291].y * h)     # Right mouth corner
    ], dtype=np.float64)
    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    success, rot_vec, _ = cv2.solvePnP(
        HEAD_POSE_MODEL_POINTS, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0, 0, 0
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles = rotation_matrix_to_euler_angles(rot_mat)
    pitch, yaw, roll = np.degrees(angles)
    return pitch, yaw, roll

def draw_face_mask(frame, landmarks, image_size):
    h, w = image_size
    face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
    points = [ (int(landmarks[idx].x * w), int(landmarks[idx].y * h)) 
               for connection in face_oval for idx in connection[:1] ]
    if points:
        pts = np.array(points, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color=(50, 50, 200))
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame

def gen_frames():
    global cap, video_streaming_active, current_metrics
    global eyes_closed_frames, blink_count, last_eye_state, fps, prev_time
    while True:
        if not video_streaming_active:
            time.sleep(0.1)
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        engaged = False
        pitch = yaw = roll = 0
        current_ear = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                frame = draw_face_mask(frame, landmarks, (h, w))
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                left_ear = compute_ear(landmarks, LEFT_EYE_IDX, (w, h))
                right_ear = compute_ear(landmarks, RIGHT_EYE_IDX, (w, h))
                current_ear = (left_ear + right_ear) / 2.0
                ear_history.append(current_ear)
                ear_avg = np.mean(ear_history)

                # Blink detection.
                if current_ear < (EAR_THRESHOLD - EAR_HYSTERESIS):
                    eyes_closed_frames += 1
                else:
                    eyes_closed_frames = 0

                if last_eye_state and current_ear < (EAR_THRESHOLD - EAR_HYSTERESIS) and eyes_closed_frames == 1:
                    blink_count += 1
                last_eye_state = (current_ear >= (EAR_THRESHOLD - EAR_HYSTERESIS))

                effective_eyes_open = (eyes_closed_frames < BLINK_FRAME_THRESHOLD)

                left_gaze, l_eye_center, _ = compute_3d_gaze(landmarks, 'left', (h, w))
                right_gaze, r_eye_center, _ = compute_3d_gaze(landmarks, 'right', (h, w))
                avg_gaze = (left_gaze + right_gaze) / 2.0
                gaze_centered = np.linalg.norm(avg_gaze) < GAZE_THRESHOLD

                norm_gaze_left = compute_normalized_horizontal_gaze(landmarks, 'left', (h, w))
                norm_gaze_right = compute_normalized_horizontal_gaze(landmarks, 'right', (h, w))
                avg_norm_gaze = (norm_gaze_left + norm_gaze_right) / 2.0

                pitch, yaw, roll = compute_head_pose(landmarks, frame)
                yaw_history.append(yaw)
                pitch_history.append(pitch)
                avg_yaw = np.mean(yaw_history)
                avg_pitch = np.mean(pitch_history)

                if effective_eyes_open:
                    if abs(avg_norm_gaze) > GAZE_HORIZONTAL_NORM_THRESHOLD:
                        engaged = False
                    elif gaze_centered:
                        engaged = True
                    else:
                        engaged = (abs(avg_yaw) <= HEAD_POSE_THRESHOLD and abs(avg_pitch) <= HEAD_POSE_THRESHOLD)
                else:
                    engaged = False

                cv2.arrowedLine(frame, tuple(np.array(l_eye_center, dtype=int)),
                                tuple(np.array(l_eye_center + left_gaze, dtype=int)), (0, 255, 0), 2)
                cv2.arrowedLine(frame, tuple(np.array(r_eye_center, dtype=int)),
                                tuple(np.array(r_eye_center + right_gaze, dtype=int)), (0, 255, 0), 2)

                cv2.putText(frame, f"Pitch: {pitch:.1f}deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}deg", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"EAR: {ear_avg:.2f}", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        else:
            engaged = False

        status_text = "Engaged" if engaged else "Not Engaged"
        status_color = (0, 255, 0) if engaged else (0, 0, 255)
        cv2.putText(frame, status_text, (w - 250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        engagement_log.append(status_text)
        current_metrics.update({
            'engagement': status_text,
            'blinks': blink_count,
            'fps': f"{fps:.1f}",
            'pitch': f"{pitch:.1f}",
            'yaw': f"{yaw:.1f}",
            'roll': f"{roll:.1f}",
            'ear': f"{ear_avg:.2f}"
        })



        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
# ---------------------------
# User session and authentication
class User(UserMixin):
    def __init__(self, user_data):
        self.user_data = user_data

    def get_id(self):
        return str(self.user_data['_id'])

@login_manager.user_loader
def load_user(user_id):
    user_data = users.find_one({'_id': ObjectId(user_id)})
    return User(user_data) if user_data else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if not all(key in request.form for key in ['email', 'password']):
            return handle_response({'error': 'Missing credentials'}, 400)

        email = request.form['email']
        password = request.form['password']
        user = users.find_one({'email': email})
        error = None
        status_code = 200

        # Artificial delay to prevent timing attacks
        _ = check_password_hash("pbkdf2:sha256:600000$dummy$dummyhash", "dummy_password")
        
        if user and check_password_hash(user['password'], password):
            if user.get('verified', False):
                login_user(User(user))
                response_data = {'redirect': url_for('dashboard')}
            else:
                error = "Account not verified. Please check your email."
                status_code = 403
        else:
            error = "Invalid email or password"
            status_code = 401

        return handle_response({'error': error} if error else response_data, status_code)
    
    return render_template('login.html')

def handle_response(data, status_code):
    if request.headers.get('Accept') == 'application/json':
        return jsonify(data), status_code
    if status_code >= 400:
        return render_template('login.html', error=data.get('error')), status_code
    return redirect(data.get('redirect', url_for('dashboard')))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.form
        errors = {}
        required_fields = ['full_name', 'email', 'password', 'user_role']
        for field in required_fields:
            if not data.get(field):
                errors[field] = f'{field.replace("_", " ").title()} is required'

        if data.get('password') != data.get('confirm_password'):
            errors['confirm_password'] = 'Passwords do not match'

        if users.find_one({'email': data.get('email')}):
            errors['email'] = 'Email already registered'
        if data.get('user_role') == 'teacher':
            teacher_id = data.get('teacher_id')
            if not teacher_id:
                errors['teacher_id'] = 'Teacher ID is required for teacher signup'
            elif not re.fullmatch(r'[A-Za-z]{5}', teacher_id):
                errors['teacher_id'] = 'Teacher ID must be exactly 5 letters'
        if errors:
            return jsonify({'errors': errors}), 400

        otp = ''.join(random.choices(string.digits, k=6))
        otp_expiry = datetime.now(timezone.utc) + timedelta(minutes=10)

        user_data = {
            'full_name': data['full_name'],
            'email': data['email'],
            'password': generate_password_hash(data['password']),
            'role': data['user_role'],
            'verified': False,
            'otp': otp,
            'otp_expiry': otp_expiry
        }
        if data.get('user_role') == 'teacher':
            user_data['teacher_pin'] = data.get('teacher_id')  # Save teacher id as teacher_pin
        
        user_id = users.insert_one(user_data).inserted_id
        send_otp_email(data['email'], otp)
        return jsonify({'redirect': url_for('verify_otp', email=data['email'])})
    
    return render_template('signup.html')

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    email = request.args.get('email', None) or request.form.get('email', None)
    if not email:
        return redirect(url_for('signup'))

    user = users.find_one({'email': email})
    if not user:
        return redirect(url_for('signup'))

    if request.method == 'POST':
        user_otp = request.form.get('otp', '').strip()
        stored_expiry = user['otp_expiry'].replace(tzinfo=timezone.utc)
        current_time = datetime.now(timezone.utc)

        print(f"\n--- OTP Verification Debug ---")
        print(f"User Email: {email}")
        print(f"DB OTP: {user['otp']} | Entered OTP: {user_otp}")
        print(f"Current Time (UTC): {current_time}")
        print(f"Stored Expiry (UTC): {stored_expiry}")

        if current_time > stored_expiry:
            new_otp = ''.join(random.choices(string.digits, k=6))
            new_expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
            users.update_one(
                {'_id': user['_id']},
                {'$set': {'otp': new_otp, 'otp_expiry': new_expiry}}
            )
            send_otp_email(email, new_otp)
            return render_template('verify_otp.html', email=email, error="OTP expired. New OTP sent to your email")

        if user_otp == user['otp']:
            update_result = users.update_one(
                {'_id': user['_id']},
                {'$set': {
                    'verified': True,
                    'otp': None,
                    'otp_expiry': None,
                    'verified_at': datetime.now(timezone.utc)
                }}
            )
            if update_result.modified_count == 1:
                verified_user = users.find_one({'_id': user['_id']})
                login_user(User(verified_user))
                print(f"--- Verification Success ---")
                print(f"Updated verified status: {verified_user['verified']}")
                return redirect(url_for('dashboard'))
            else:
                return render_template('verify_otp.html', email=email, error="Verification failed. Please try again")

        return render_template('verify_otp.html', email=email, error="Invalid OTP. Check your email")
    
    return render_template('verify_otp.html', email=email)


@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.user_data['role'] == 'teacher':
        active_meetings = list(meetings.find({
            'teacher_id': current_user.get_id(),
            'active': True
        }))
        # For each meeting, look up the participants' details and aggregate engagement metrics.
        for meeting in active_meetings:
            student_list = []
            for student_id in meeting.get('participants', []):
                student = users.find_one({'_id': ObjectId(student_id)})
                if student:
                    # Get session records for this student and meeting.
                    student_sessions = list(sessions.find({
                        'user_id': student_id,
                        'meeting_code': meeting['code']
                    }))
                    if student_sessions:
                        engaged_count = sum(1 for sess in student_sessions if sess.get('engagement') == 'Engaged')
                        overall_engagement = (engaged_count / len(student_sessions)) * 100
                        overall_engagement = round(overall_engagement, 1)
                    else:
                        overall_engagement = None
                    student_list.append({
                        'full_name': student.get('full_name', ''),
                        'email': student.get('email', ''),
                        'engagement': overall_engagement
                    })
            meeting['students'] = student_list
        return render_template('teacher_dashboard.html', meetings=active_meetings)
    else:
        return render_template('student_dashboard.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict_emotion():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_str = data['image']
        if ',' in image_str:
            header, encoded = image_str.split(',', 1)
        else:
            encoded = image_str
        image_data = base64.b64decode(encoded)

        image = Image.open(BytesIO(image_data)).convert('L')
        image = image.resize((48, 48))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        predictions = model.predict(img_array)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_idx = int(np.argmax(predictions))
        confidence = float(predictions[0][emotion_idx])

        sessions.insert_one({
            'user_id': current_user.get_id(),
            'meeting_code': data.get('meeting_code', ''),
            'emotion': emotion_labels[emotion_idx],
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc)
        })

        return jsonify({
            'emotion': emotion_labels[emotion_idx],
            'confidence': confidence
        })
    
    except Exception as e:
        print("Error in predict_emotion:", e)
        return jsonify({'error': str(e)}), 500

def send_otp_email(to_email, otp):
    msg = EmailMessage()
    msg.set_content(f'Your verification OTP is: {otp}')
    msg['Subject'] = 'E LEARN Verification Code'
    msg['From'] = config['EMAIL_USER']
    msg['To'] = to_email
    with smtplib.SMTP_SSL(config['SMTP_SERVER'], config['SMTP_PORT']) as server:
        server.login(config['EMAIL_USER'], config['EMAIL_PASSWORD'])
        server.send_message(msg)

def send_meeting_email(to_email, meeting_code, action):
    if action == "start":
        subject = "Meeting Started"
        content = f"A meeting has started. Your meeting code is: {meeting_code}"
    else:
        subject = "Meeting Terminated"
        content = "The meeting has been terminated."
    
    msg = EmailMessage()
    msg.set_content(content)
    msg['Subject'] = subject
    msg['From'] = config['EMAIL_USER']
    msg['To'] = to_email
    with smtplib.SMTP_SSL(config['SMTP_SERVER'], config['SMTP_PORT']) as server:
        server.login(config['EMAIL_USER'], config['EMAIL_PASSWORD'])
        server.send_message(msg)

@app.route('/mobile-login', methods=['GET', 'POST'])
def mobile_login():
    if request.method == 'POST':
        phone = request.form.get('phone')
        otp_code = ''.join(random.choices(string.digits, k=6))
        otps.insert_one({
            'phone': phone,
            'otp': otp_code,
            'expires_at': datetime.now(timezone.utc) + timedelta(minutes=5)
        })
        print(f"DEBUG: OTP {otp_code} sent to {phone}")
        return redirect(url_for('verify_otp', phone=phone))
    
    return render_template('mobile_login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = users.find_one({'email': email})
        if user:
            reset_token = secrets.token_urlsafe(32)
            reset_expires = datetime.now(timezone.utc) + timedelta(hours=1)
            users.update_one(
                {'_id': user['_id']},
                {'$set': {'reset_token': reset_token, 'reset_expires': reset_expires}}
            )
            reset_link = url_for('reset_password', token=reset_token, _external=True)
            msg = EmailMessage()
            msg.set_content(f"Reset your password: {reset_link}")
            msg['Subject'] = 'Password Reset Request'
            msg['From'] = config['EMAIL_USER']
            msg['To'] = email
            with smtplib.SMTP_SSL(config['SMTP_SERVER'], config['SMTP_PORT']) as server:
                server.login(config['EMAIL_USER'], config['EMAIL_PASSWORD'])
                server.send_message(msg)
        return render_template('forgot_password.html', message="If an account exists, you'll receive a reset email")
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = users.find_one({
        'reset_token': token,
        'reset_expires': {'$gt': datetime.now(timezone.utc)}
    })
    if not user:
        return render_template('reset_password.html', error="Invalid or expired token")
    
    if request.method == 'POST':
        new_password = request.form.get('password')
        users.update_one(
            {'_id': user['_id']},
            {'$set': {
                'password': generate_password_hash(new_password),
                'reset_token': None,
                'reset_expires': None
            }}
        )
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

@app.route('/generate-meeting', methods=['POST'])
@login_required
def generate_meeting():
    if current_user.user_data['role'] != 'teacher':
        return jsonify({'error': 'Unauthorized'}), 403
    meeting_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    meeting_data = {
        'code': meeting_code,
        'teacher_id': current_user.get_id(),
        'created_at': datetime.now(timezone.utc),
        'participants': [],
        'active': True
    }
    meetings.insert_one(meeting_data)
    return jsonify({'code': meeting_code})

@app.route('/start-meeting', methods=['POST'])
@login_required
def start_meeting():
    if current_user.user_data['role'] != 'teacher':
        return jsonify({'error': 'Unauthorized'}), 403
    meeting_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    meeting_data = {
        'code': meeting_code,
        'teacher_id': current_user.get_id(),
        'created_at': datetime.now(timezone.utc),
        'participants': [],
        'active': True
    }
    meetings.insert_one(meeting_data)
    all_users = list(users.find({}))
    for user in all_users:
        send_meeting_email(user['email'], meeting_code, "start")
    return jsonify({'success': True, 'meetingCode': meeting_code})

@app.route('/join-meeting', methods=['POST'])
@login_required
def join_meeting():
    code = request.form.get('code')
    teacher_pin_input = request.form.get('teacher_pin')
    if not teacher_pin_input:
        return render_template('student_dashboard.html', error='Teacher PIN is required to join the meeting.')
    meeting = meetings.find_one({'code': code, 'active': True})
    if not meeting:
        return render_template('student_dashboard.html', error='Invalid meeting code or meeting is not active.')
    teacher = users.find_one({'_id': ObjectId(meeting['teacher_id'])})
    if not teacher or teacher.get('teacher_pin') != teacher_pin_input:
        return render_template('student_dashboard.html', error='Invalid Teacher PIN.')
    
    meetings.update_one(
        {'_id': meeting['_id']},
        {'$addToSet': {'participants': current_user.get_id()}}
    )
    
    # Send join email notification to teacher.
    student_name = current_user.user_data.get('full_name', 'Unknown')
    student_email = current_user.user_data.get('email', 'Unknown')
    engagement = None  # Engagement not computed yet at join time.
    send_student_join_email(
        teacher_email=teacher['email'],
        student_email=student_email,
        student_name=student_name,
        meeting_code=code,
        engagement=engagement
    )
    
    return redirect(url_for('classroom', code=code))


def send_student_join_email(teacher_email, student_email, student_name, meeting_code, engagement=None):
    engagement_str = f"{engagement}%" if engagement is not None else "N/A"
    
    msg = EmailMessage()
    msg.set_content(
        f"A new student has joined your meeting (Code: {meeting_code}).\n\n"
        f"Student Name: {student_name}\n"
        f"Student Email: {student_email}\n"
        f"Overall Engagement: {engagement_str}"
    )
    msg['Subject'] = 'New Student Joined Your Meeting'
    msg['From'] = config['EMAIL_USER']
    msg['To'] = teacher_email
    with smtplib.SMTP_SSL(config['SMTP_SERVER'], config['SMTP_PORT']) as server:
         server.login(config['EMAIL_USER'], config['EMAIL_PASSWORD'])
         server.send_message(msg)



@app.route('/stop-meeting', methods=['POST'])
@login_required
def stop_meeting():
    if current_user.user_data['role'] != 'teacher':
        return jsonify({'error': 'Unauthorized'}), 403
    meeting = meetings.find_one({
        'teacher_id': current_user.get_id(),
        'active': True
    })
    if meeting is None:
        return jsonify({'error': 'No active meeting found'}), 404
    meetings.update_one({'_id': meeting['_id']}, {'$set': {'active': False}})
    all_users = list(users.find({}))
    for user in all_users:
        send_meeting_email(user['email'], meeting['code'], "stop")
    return jsonify({'success': True})

# New video streaming endpoint
@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classroom/<code>')
@login_required
def classroom(code):
    # Render classroom page and pass the meeting code
    return render_template('classroom.html', code=code)

@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        emit('user_status', {'userId': current_user.get_id(), 'online': True}, broadcast=True)

@socketio.on('start_class')
def handle_start_class(data):
    code = data['code']
    emit('class_started', {'code': code}, room=code, broadcast=True)

@socketio.on('join_classroom')
def handle_join_classroom(data):
    code = data['code']
    meetings.update_one(
        {'code': code},
        {'$addToSet': {'participants': current_user.get_id()}}
    )
    meeting = meetings.find_one({'code': code})
    participant_count = len(meeting.get('participants', []))
    emit('participant_update', {'count': participant_count}, room=code)

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%H:%M %d-%b'):
    return value.strftime(format)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/resend-otp/<email>')
def resend_otp(email):
    user = users.find_one({'email': email})
    if not user:
        return redirect(url_for('signup'))
    new_otp = ''.join(random.choices(string.digits, k=6))
    new_expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
    users.update_one(
        {'_id': user['_id']},
        {'$set': {'otp': new_otp, 'otp_expiry': new_expiry}}
    )
    send_otp_email(email, new_otp)
    return redirect(url_for('verify_otp', email=email))

@app.route('/stop_camera', methods=['POST'])
@login_required
def stop_camera():
    global video_streaming_active, cap
    if cap is not None:
        cap.release()
        cap = None
    video_streaming_active = False
    return jsonify({'status': 'stopped'})

@app.route('/start_camera', methods=['POST'])
@login_required
def start_camera():
    global video_streaming_active, cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_streaming_active = True
    return jsonify({'status': 'started'})

@app.route('/metrics')
@login_required
def metrics():
    return jsonify(current_metrics)

@app.route('/export-pdf/<string:meeting_code>')
@login_required
def export_pdf(meeting_code):
    """
    Generates a PDF report for the given meeting_code, using ReportLab.
    """
    # 1) Verify the meeting belongs to this teacher
    meeting = meetings.find_one({
        'code': meeting_code,
        'teacher_id': current_user.get_id()
    })
    if not meeting:
        return "Unauthorized or meeting not found", 403

    # 2) Fetch all session records for this meeting, sorted by timestamp:
    cursor = sessions.find({'meeting_code': meeting_code}).sort('timestamp', 1)

    # Build a list of rows (header + each record)
    data_rows = []
    data_rows.append(['Timestamp (UTC)', 'Student', 'Engagement', 'Confidence'])
    engaged_count = 0
    total_count = 0

    for rec in cursor:
        total_count += 1
        if rec.get('engagement') == 'Engaged':
            engaged_count += 1

        student_doc = users.find_one({'_id': ObjectId(rec['user_id'])})
        student_name = student_doc.get('full_name', 'Unknown')
        engagement_str = rec.get('engagement', 'N/A')
        confidence_val = rec.get('confidence', 0.0)
        confidence_pct = f"{confidence_val * 100:.1f}%"

        ts_str = rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        data_rows.append([ts_str, student_name, engagement_str, confidence_pct])

    overall_engagement = round((engaged_count / total_count) * 100, 1) if total_count else 0.0

    # 3) Build the PDF in memory with ReportLab
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    style_title = styles['Heading1']
    style_sub = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontSize=12,
        leading=14,
        spaceAfter=12
    )

    elements = []

    # Title
    title = Paragraph(f"Meeting Report: {meeting_code}", style_title)
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Meeting Info
    created_at_str = meeting['created_at'].strftime('%d %b %Y, %H:%M') + " UTC"
    info_lines = [
        f"<strong>Teacher:</strong> {current_user.user_data['full_name']} ({current_user.user_data['email']})",
        f"<strong>Created On:</strong> {created_at_str}",
        f"<strong>Overall Engagement %:</strong> {overall_engagement}%",
        f"<strong>Total Records:</strong> {total_count}"
    ]
    for line in info_lines:
        elements.append(Paragraph(line, style_sub))
    elements.append(Spacer(1, 12))

    # Table of all session records
    tbl_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),    # center “Engagement” column
        ('ALIGN', (3, 1), (3, -1), 'RIGHT'),     # right-align “Confidence”
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
    ])

    table = Table(data_rows, colWidths=[120, 120, 100, 80])
    table.setStyle(tbl_style)
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)

    # 4) Send PDF back to browser as a file download
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"report_{meeting_code}.pdf",
        mimetype="application/pdf"
    )

@app.route('/exportresult', methods=['GET', 'POST'])
@login_required
def exportresult():
    # Calculate overall engagement percentage from the engagement_log.
    if engagement_log:
        engaged_count = sum(1 for state in engagement_log if state == "Engaged")
        overall_engagement = (engaged_count / len(engagement_log)) * 100
    else:
        overall_engagement = 0.0

    # Get additional metrics from current_metrics, or default to 'N/A'.
    noise_level = current_metrics.get('noise_level', 'N/A')
    mouse_clicks = current_metrics.get('mouse_clicks', 'N/A')

    if request.method == 'POST':
        report = f"""Class Report:
Engagement: {current_metrics.get('engagement', 'N/A')}
Overall Engagement: {overall_engagement:.1f}%
Blinks: {current_metrics.get('blinks', 'N/A')}
FPS: {current_metrics.get('fps', 'N/A')}
Head Pose: Pitch: {current_metrics.get('pitch', 'N/A')} deg, Yaw: {current_metrics.get('yaw', 'N/A')} deg, Roll: {current_metrics.get('roll', 'N/A')} deg
EAR: {current_metrics.get('ear', 'N/A')}
Noise Level: {noise_level}
Mouse Clicks: {mouse_clicks}
"""
        teacher_email = current_user.user_data['email']
        msg = EmailMessage()
        msg.set_content(report)
        msg['Subject'] = 'Class Report'
        msg['From'] = config['EMAIL_USER']
        msg['To'] = teacher_email
        try:
            with smtplib.SMTP_SSL(config['SMTP_SERVER'], config['SMTP_PORT']) as server:
                server.login(config['EMAIL_USER'], config['EMAIL_PASSWORD'])
                server.send_message(msg)
            return jsonify({'status': 'Report sent successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Render exportresult.html for GET requests with additional metrics.
    return render_template('exportresult.html',
                           overall_engagement=f"{overall_engagement:.1f}",
                           blinks=current_metrics.get('blinks', 'N/A'),
                           fps=current_metrics.get('fps', 'N/A'),
                           pitch=current_metrics.get('pitch', 'N/A'),
                           yaw=current_metrics.get('yaw', 'N/A'),
                           roll=current_metrics.get('roll', 'N/A'),
                           ear=current_metrics.get('ear', 'N/A'),
                           noise_level=noise_level,
                           mouse_clicks=mouse_clicks)

@app.route('/update_engagement', methods=['POST'])
@login_required
def update_engagement():
    data = request.json
    engagement_state = data.get('engagement', 'Inactive')
    meeting_code = data.get('meeting_code')  # The front-end must send the code

    # Insert a record into sessions
    sessions.insert_one({
        'user_id': current_user.get_id(),
        'meeting_code': meeting_code,
        'engagement': 'Engaged' if engagement_state == 'Engaged' else 'Not Engaged',
        'timestamp': datetime.now(timezone.utc)
    })

    return jsonify({"status": "Engagement updated"})
@app.route('/voice_metrics', methods=['GET', 'POST'])
def voice_metrics():
    if request.method == 'POST':
        # Here, you might receive noise level data from the client.
        noise_level = request.json.get('noise')
        # Process or store the noise level data as needed.
        # For this example, we simply acknowledge receipt.
        return jsonify({"status": "Noise level received", "noise": noise_level})
    else:
        # For GET requests, you could return a default value or the latest noise level.
        return jsonify({"noise": "N/A"})
@app.route('/leave-meeting', methods=['POST'])
@login_required

def leave_meeting():
    code = request.form.get('code')
    meeting = meetings.find_one({'code': code})
    if not meeting:
        return jsonify({'error': 'Meeting not found'}), 404

    # Remove the student from the meeting's participants.
    meetings.update_one(
        {'_id': meeting['_id']},
        {'$pull': {'participants': current_user.get_id()}}
    )
    
    # Calculate overall engagement for this student in the meeting.
    student_sessions = list(sessions.find({
        'user_id': current_user.get_id(),
        'meeting_code': code
    }))
    if student_sessions:
        engaged_count = sum(1 for sess in student_sessions if sess.get('engagement') == 'Engaged')
        overall_engagement = round((engaged_count / len(student_sessions)) * 100, 1)
    else:
        overall_engagement = None

    teacher = users.find_one({'_id': ObjectId(meeting['teacher_id'])})
    student_name = current_user.user_data.get('full_name', 'Unknown')
    student_email = current_user.user_data.get('email', 'Unknown')
    
    # Send leave email notification to teacher with overall engagement.
    send_student_leave_email(
        teacher_email=teacher['email'],
        student_email=student_email,
        student_name=student_name,
        meeting_code=code,
        engagement=overall_engagement
    )
    
    return jsonify({'status': 'left meeting successfully'})

def send_student_leave_email(teacher_email, student_email, student_name, meeting_code, engagement=None):
    engagement_str = f"{engagement}%" if engagement is not None else "N/A"
    msg = EmailMessage()
    msg.set_content(
        f"Student Left Meeting Report:\n\n"
        f"Meeting Code: {meeting_code}\n"
        f"Student Name: {student_name}\n"
        f"Student Email: {student_email}\n"
        f"Overall Engagement: {engagement_str}"
    )
    msg['Subject'] = f'Student Left Meeting: {student_name}'
    msg['From'] = config['EMAIL_USER']
    msg['To'] = teacher_email
    try:
        with smtplib.SMTP_SSL(config['SMTP_SERVER'], config['SMTP_PORT']) as server:
            server.login(config['EMAIL_USER'], config['EMAIL_PASSWORD'])
            server.send_message(msg)
    except Exception as e:
        print("Error sending leave email:", e)
from bson.son import SON
from flask import jsonify

@app.route('/api/meeting-engagement/<string:code>')
@login_required
def meeting_engagement(code):
    """
    Returns JSON data for Chart.js: { timestamps: [...], engagement_percent: [...] }
    We group all session documents for this meeting by minute (or hour), and compute
    engagement percentage per bucket.
    """
    # 1. Ensure the meeting actually belongs to this teacher (optional, but recommended)
    m = meetings.find_one({'code': code, 'teacher_id': current_user.get_id()})
    if not m:
        return jsonify({'error': 'Meeting not found or unauthorized'}), 404

    # 2. Aggregate in Mongo: group by hour/minute of timestamp (UTC)
    pipeline = [
        {'$match': {'meeting_code': code}},
        {'$group': {
            '_id': {
                'hour': {'$hour': {'date': '$timestamp', 'timezone': 'UTC'}},
                'minute': {'$minute': {'date': '$timestamp', 'timezone': 'UTC'}}
            },
            'total': {'$sum': 1},
            'engagedCount': {
                '$sum': {
                    '$cond': [{'$eq': ['$engagement', 'Engaged']}, 1, 0]
                }
            }
        }},
        {'$sort': SON([('_id.hour', 1), ('_id.minute', 1)])}
    ]

    agg_results = list(sessions.aggregate(pipeline))
    labels = []
    percents = []
    for grp in agg_results:
        h = grp['_id']['hour']
        mnt = grp['_id']['minute']
        label = f"{h:02d}:{mnt:02d}"
        pct = round((grp['engagedCount'] / grp['total']) * 100, 1)
        labels.append(label)
        percents.append(pct)

    return jsonify({'timestamps': labels, 'engagement_percent': percents})


if __name__ == '__main__':
    socketio.run(app, debug=True)


