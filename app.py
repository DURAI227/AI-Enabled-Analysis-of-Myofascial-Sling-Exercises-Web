
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from flask_cors import CORS
from ultralytics import YOLO
import math
import time
import json
import os
from supabase import create_client, Client

app = Flask(__name__)
app.secret_key = 'super_secret_key_physio_ai' # Change this in production
CORS(app)

# Supabase Setup - PLEASE REPLACE WITH YOUR DETAILS
SUPABASE_URL = "https://tqypautmlmjjasvqwdgw.supabase.co"
SUPABASE_KEY = "sb_publishable_ALUruq3-ztjt48v46ZqiQw_-sZw4SHw"

try:
    from supabase import create_client, Client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase connection failed: {e}. Using mock client.")
    class MockSupabase:
        def __init__(self):
            self.auth = self.Auth()
            self._data = {} # Simple in-memory storage for tables
            
        class Auth:
            def sign_in_with_password(self, params):
                class User: id = "mock_user_id"; email = "test@example.com"
                class Response: user = User()
                return Response()
            def sign_up(self, params):
                class User: id = "mock_user_id"; email = "test@example.com"
                class Response: user = User()
                return Response()
            def sign_out(self): pass
        
        def table(self, table_name):
            outer_self = self
            class Query:
                def __init__(self, t_name):
                    self.t_name = t_name
                    if t_name not in outer_self._data:
                        outer_self._data[t_name] = []
                
                def select(self, *args): return self
                def insert(self, data):
                    if isinstance(data, list):
                        outer_self._data[self.t_name].extend(data)
                    else:
                        outer_self._data[self.t_name].append(data)
                    return self
                def update(self, *args): return self
                def eq(self, *args): return self
                def order(self, *args, **kwargs): return self
                def execute(self):
                    class Result: 
                        def __init__(self, d):
                            self.data = d
                    return Result(outer_self._data[self.t_name])
            return Query(table_name)
    supabase = MockSupabase()

# Load YOLOv8 Pose Model - Using Medium for better performance/accuracy balance
model = YOLO('yolov8m-pose.pt')

# Exercise Database with detailed instructions
EXERCISES = [
    {
        "id": 1,
        "name": "Bird Dog",
        "duration": 45,
        "target_reps": 10,
        "instructions": "Position yourself in tabletop. Extend opposite arm and leg simultaneously while maintaining a stable spine.",
        "animation_type": "bird_dog",
        "target_muscles": ["Core", "Glutes", "Shoulders"],
        "image": "/static/exercise_images/Gemini_Generated_Image_pa9ry7pa9ry7pa9r.png",
        "video": "/static/animations/Video/bird_dog.mp4"
    },
    {
        "id": 2,
        "name": "Clamshells",
        "duration": 30,
        "target_reps": 12,
        "instructions": "Lie on your side, knees bent. Lift top knee while keeping feet together to activate hip stabilizers.",
        "animation_type": "clamshells",
        "target_muscles": ["Glutes", "Hips", "Lateral Sling"],
        "image": "/static/exercise_images/Gemini_Generated_Image_1kjaoe1kjaoe1kja.png",
        "video": "/static/animations/Video/clamshells.mp4"
    },
    {
        "id": 3,
        "name": "Forward Lunge",
        "duration": 30,
        "target_reps": 10,
        "instructions": "Step forward into a lunge. Keep torso upright and core engaged to activate the anterior power sling.",
        "animation_type": "forward_lunge",
        "target_muscles": ["Quadriceps", "Glutes", "Core"],
        "image": "/static/exercise_images/Gemini_Generated_Image_apse0lapse0lapse.png",
        "video": "/static/animations/Video/forward_lunge.mp4"
    },
    {
        "id": 4,
        "name": "Good Morning",
        "duration": 30,
        "target_reps": 15,
        "instructions": "Hinge at the hips with a slight knee bend. Keep your back flat to stretch the posterior chain.",
        "animation_type": "good_morning",
        "target_muscles": ["Hamstrings", "Lower Back", "Glutes"],
        "image": "/static/exercise_images/Gemini_Generated_Image_r8kxlar8kxlar8kx.png",
        "video": "/static/animations/Video/good_morning.mp4"
    },
    {
        "id": 5,
        "name": "Standing Hamstring Stretch",
        "duration": 30,
        "target_reps": 5,
        "instructions": "Stand and reach for your toes while keeping legs straight. Focus on the deep longitudinal sling.",
        "animation_type": "forward_bend",
        "target_muscles": ["Hamstrings", "Lower Back"],
        "image": "/static/exercise_images/Gemini_Generated_Image_lupojelupojelupo - Copy.png",
        "video": "/static/animations/Video/hamstring_stretch.mp4"
    },
    {
        "id": 6,
        "name": "Hip Flexor Stretch",
        "duration": 30,
        "target_reps": 5,
        "instructions": "Step into a half-kneeling position. Push hips forward gently to stretch the front hip.",
        "animation_type": "hip_flexor",
        "target_muscles": ["Hip Flexors", "Quads"],
        "image": "/static/exercise_images/Gemini_Generated_Image_6jmh5u6jmh5u6jmh - Copy.png",
        "video": "/static/animations/Video/hip_flexor.mp4"
    },
    {
        "id": 7,
        "name": "Lateral Lunge",
        "duration": 30,
        "target_reps": 10,
        "instructions": "Step to the side and lower your hips. Keep the other leg straight.",
        "animation_type": "lateral_lunge",
        "target_muscles": ["Adductors", "Glutes", "Hips"],
        "image": "/static/exercise_images/Gemini_Generated_Image_w1xcf4w1xcf4w1xc.png",
        "video": "/static/animations/Video/lateral_lunge.mp4"
    },
    {
        "id": 8,
        "name": "Marching",
        "duration": 30,
        "target_reps": 20,
        "instructions": "Stand tall and march in place, bringing knees high. Maintain core stability.",
        "animation_type": "marching",
        "target_muscles": ["Hip Flexors", "Core"],
        "image": "/static/exercise_images/Gemini_Generated_Image_ukid8aukid8aukid.png",
        "video": "/static/animations/Video/marching.mp4"
    },
    {
        "id": 9,
        "name": "Pallof Press",
        "duration": 30,
        "target_reps": 10,
        "instructions": "Hold a band at chest height and press forward, resisting the pull to the side.",
        "animation_type": "pallof_press",
        "target_muscles": ["Obliques", "Core", "Shoulders"],
        "image": "/static/exercise_images/Gemini_Generated_Image_i7ztlmi7ztlmi7zt.png",
        "video": "/static/animations/Video/pallof_press.mp4"
    },
    {
        "id": 10,
        "name": "Reverse Lunge",
        "duration": 30,
        "target_reps": 10,
        "instructions": "Step backward into a lunge, maintaining balance and upright posture.",
        "animation_type": "reverse_lunge",
        "target_muscles": ["Glutes", "Quadriceps", "Core"],
        "image": "/static/exercise_images/Gemini_Generated_Image_8strjw8strjw8str.png",
        "video": "/static/animations/Video/reverse_lunge.mp4"
    },
    {
        "id": 11,
        "name": "Single Leg Bridge",
        "duration": 30,
        "target_reps": 12,
        "instructions": "Lie on back, lift one leg. Push through the other heel to lift hips.",
        "animation_type": "single_leg_bridge",
        "target_muscles": ["Glutes", "Hamstrings", "Core"],
        "image": "/static/exercise_images/Gemini_Generated_Image_xni2boxni2boxni2.png",
        "video": "/static/animations/Video/single_leg.mp4"
    },
    {
        "id": 12,
        "name": "Standing Cable Chop",
        "duration": 30,
        "target_reps": 10,
        "instructions": "Pull diagonally across your body from high to low. Rotate your torso.",
        "animation_type": "cable_chop",
        "target_muscles": ["Obliques", "Core", "Shoulders"],
        "image": "/static/exercise_images/Gemini_Generated_Image_xrj1oixrj1oixrj1.png",
        "video": "/static/animations/Video/standing_cable_chop.mp4"
    },
    {
        "id": 13,
        "name": "Trunk Rotation",
        "duration": 30,
        "target_reps": 15,
        "instructions": "Rotate your torso from side to side while keeping your hips stable.",
        "animation_type": "rotation",
        "target_muscles": ["Obliques", "Spine", "Core"],
        "image": "/static/exercise_images/Gemini_Generated_Image_hp48wehp48wehp48.png"
    }
]


class BiomechState:
    """Professional State Manager"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.phase = "IDLE"  # IDLE, TEST1, EXERCISE, TEST2, COMPLETE
        self.sub_phase = "READY"
        self.is_active = True
        self.test1_score = 0
        self.test2_score = 0
        self.test1_status = None
        self.exercise_index = 0
        self.max_exercises = 0
        self.current_exercise = None
        self.analyzer_cache = {} # Cache for exercise-specific analyzers
        
        self.metrics = {
            "flexibility_score": 0,
            "hip_angle": 180,
            "knee_bend_warning": False,
            "alignment_check": False,
            "feedback": "Step into the center of the frame.",
            "phase": "IDLE",
            "exercise_progress": "0/0",
            "assessment_name": "Toe-Touch Analysis"
        }

state = BiomechState()


class BiomechAnalyzer:
    """Professional Biomechanical Analysis for Toe Touch"""
    def __init__(self):
        self.prev_time = time.time()
        self.knee_history = []
        self.landmark_history = []
        self.max_flex = 0
        self.min_angle = 180

    def reset(self):
        self.max_flex = 0
        self.min_angle = 180
        self.knee_history = []
        self.landmark_history = []

    def get_angle(self, p1, p2, p3):
        """Angle at p2 using points p1-p2-p3"""
        if any(v is None or len(v) < 2 or v[0] == 0 for v in [p1, p2, p3]): 
            return 180
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0: return 180
        cosine_angle = np.dot(v1, v2) / (n1 * n2 + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def analyze_frame(self, frame):
        # RUN ON RAW DATA (Not Flipped)
        results = model(frame, verbose=False)
        annotated_frame = frame.copy()
        
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kpts = results[0].keypoints.xy.cpu().numpy()[0]
            confs = results[0].keypoints.conf.cpu().numpy()[0] if results[0].keypoints.conf is not None else np.ones(17)
            
            # Select most visible profile
            r_idx, l_idx = [6, 12, 14, 16], [5, 11, 13, 15]
            r_conf, l_conf = np.mean(confs[r_idx]), np.mean(confs[l_idx])
            side = "right" if r_conf > l_conf else "left"
            idx = r_idx if side == "right" else l_idx
            s, h, k, a = kpts[idx[0]], kpts[idx[1]], kpts[idx[2]], kpts[idx[3]]
            
            # 1. SIDE PROFILE GEOMETRY CHECK
            # We are sideways when torso width (l-r) is much smaller than torso length (s-h)
            torso_width = abs(kpts[5][0] - kpts[6][0])
            torso_length = abs(s[1] - h[1]) + 1e-6
            is_sideways = torso_width < (torso_length * 0.75)
            
            # 2. METRICS
            hip_angle = self.get_angle(s, h, k)
            knee_angle_raw = self.get_angle(h, k, a)
            
            # 3. KNEE SMOOTHING
            self.knee_history.append(knee_angle_raw)
            if len(self.knee_history) > 10: self.knee_history.pop(0)
            knee_angle = np.mean(self.knee_history)
            is_knee_bent = knee_angle < 158 # Threshold for "locked" knee
            
            # 4. SCORE MAPPING
            # Baseline: 175deg=0%, Max Bend: 45deg=100%
            current_flex = max(0, min(100, int((175 - hip_angle) / (175 - 45) * 100)))
            
            # workflow tracking
            if state.phase in ["TEST1", "TEST2"] and state.sub_phase == "IN_PROGRESS":
                if not is_knee_bent and is_sideways:
                    if current_flex > self.max_flex:
                        self.max_flex = current_flex
                        self.min_angle = hip_angle
            
            # Feedback Logic
            if not is_sideways:
                feedback = "ALIGNMENT: Turn sideways to camera."
            elif is_knee_bent:
                feedback = "KNEE LOCK: Keep your legs straight!"
            elif state.phase in ["TEST1", "TEST2"]:
                feedback = f"TESTING: Bend deep! Current: {current_flex}% | Max: {self.max_flex}%"
            else:
                feedback = "READY: Click Start Assessment." if is_sideways else "Turn sideways."

            state.metrics.update({
                "flexibility_score": current_flex,
                "hip_angle": int(hip_angle),
                "knee_bend_warning": bool(is_knee_bent),
                "alignment_check": bool(is_sideways),
                "feedback": feedback,
            })
            
            # Draw HUD - Only if confidence is high enough
            color = (0, 255, 0) if not is_knee_bent else (0, 0, 255)
            for i in range(len(idx)-1):
                pk1, pk2 = idx[i], idx[i+1]
                if confs[pk1] > 0.5 and confs[pk2] > 0.5:
                    p1 = (int(kpts[pk1][0]), int(kpts[pk1][1]))
                    p2 = (int(kpts[pk2][0]), int(kpts[pk2][1]))
                    cv2.line(annotated_frame, p1, p2, color, 4)
                
            # Vertical Progress Bar - Removed as requested
            # h_v, w_v = frame.shape[:2]
            # cv2.rectangle(annotated_frame, (w_v-30, int(h_v*0.2)), (w_v-10, int(h_v*0.8)), (40,40,40), -1)
            # bar_h = int((current_flex/100) * (h_v*0.6))
            # cv2.rectangle(annotated_frame, (w_v-30, int(h_v*0.8)-bar_h), (w_v-10, int(h_v*0.8)), (0,255,120), -1)

        return annotated_frame

analyzer = BiomechAnalyzer()


def gen_frames():
    """Video feed generator"""
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        # Determine which analysis to run
        curr_ex = state.current_exercise
        if state.phase == "EXERCISE" and curr_ex:
            import analysis_poses
            anim_type = curr_ex.get('animation_type')
            cached_analyzer = state.analyzer_cache.get(anim_type)
            
            # Run specialized exercise analysis
            frame, metrics, feedback, new_analyzer = analysis_poses.exercises(
                anim_type, frame, cached_analyzer
            )
            
            # Save analyzer back to cache for persistence (rep counts, state)
            if new_analyzer:
                state.analyzer_cache[anim_type] = new_analyzer
            
            # Update global state metrics for frontend
            if metrics:
                state.metrics.update(metrics)
            
            # Update labels for frontend
            state.metrics['assessment_name'] = curr_ex.get('name', 'Exercise')
            state.metrics['target_reps'] = curr_ex.get('target_reps', 10)
            state.metrics['feedback'] = feedback
            state.metrics['phase'] = "EXERCISE"
        else:
            # Default to Toe-Touch assessment
            frame = analyzer.analyze_frame(frame)
            state.metrics['assessment_name'] = "Toe-Touch Analysis"
            state.metrics['phase'] = state.phase
        
        frame = cv2.flip(frame, 1)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ============== ROUTES ==============

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')

@app.route('/assessment')
def assessment_page():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('assessment.html')

@app.route('/login', methods=['GET'])
def login_page():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        session['user'] = response.user.email
        return jsonify({"message": "Login successful"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    try:
        # Sign up
        response = supabase.auth.sign_up({"email": email, "password": password})
        
        # Check if user is created (and possibly auto-confirmed if disabled in settings)
        if response.user:
            # Auto login by setting session
            session['user'] = response.user.email
            return jsonify({"message": "Registration successful. Logging you in..."})
            
        return jsonify({"message": "Please check your email for confirmation link."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/logout')
def logout():
    session.pop('user', None)
    supabase.auth.sign_out()
    return redirect(url_for('login_page'))


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/metrics')
def get_metrics():
    # Safe serialization for numpy types
    safe_metrics = {k: (v.item() if hasattr(v, 'item') else v) for k, v in state.metrics.items()}
    return jsonify(safe_metrics)


@app.route('/exercises')
def get_exercises():
    """Return all available exercises"""
    return jsonify(EXERCISES)


@app.route('/history')
def history_page():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('history.html')


@app.route('/api/history')
def get_history_api():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        # Fetching from 'assessments' table
        response = supabase.table('assessments').select("*").execute()
        # Sort by created_at descending if possible, or just reverse
        data = sorted(response.data, key=lambda x: x.get('created_at', ''), reverse=True)
        return jsonify(data)
    except Exception as e:
        print(f"History fetch error: {e}")
        return jsonify([])


@app.route('/start_assessment', methods=['POST'])
def start_assessment():
    """Start the assessment workflow - begins Test 1"""
    data = request.json or {}
    assessment_type = data.get('type', 'toe_touch')
    
    if state.phase == "IDLE":
        state.reset()
        state.phase = "TEST1"
        state.sub_phase = "IN_PROGRESS"
        
        analyzer.reset()
        state.metrics["assessment_name"] = "Toe-Touch Analysis"
        state.metrics["feedback"] = "Stand sideways to the camera."

        return jsonify({
            "status": "started",
            "phase": "TEST1",
            "message": f"{state.metrics['assessment_name']} started!"
        })
    return jsonify({"status": "error", "message": "Assessment already in progress"}), 400


@app.route('/action', methods=['POST'])
def handle_action():
    data = request.json
    action = data.get('action')
    
    if action == "START_TEST1":
        state.phase = "TEST1"
        state.sub_phase = "IN_PROGRESS"
        analyzer.reset()
        return jsonify({"status": "started", "phase": "TEST1"})
        
    elif action == "END_TEST1":
        state.test1_score = analyzer.max_flex
        state.test1_status = "PASS" if state.test1_score >= 70 else "FAIL"
        state.max_exercises = 5 if state.test1_status == "PASS" else 10
        state.exercise_index = 0
        state.phase = "EXERCISE"
        state.sub_phase = "IN_PROGRESS"
        
        # Filter exercises (exclude neck for now as per frontend)
        neck_keywords = ['neck', 'trapezius', 'cat-cow']
        filtered_ex = [ex for ex in EXERCISES if not any(k in ex['name'].lower() for k in neck_keywords)]
        if filtered_ex:
            state.current_exercise = filtered_ex[0]
            
        return jsonify({
            "score": state.test1_score,
            "result": state.test1_status,
            "count": state.max_exercises,
            "exercise": state.current_exercise
        })
        
    elif action == "NEXT_EX":
        state.exercise_index += 1
        neck_keywords = ['neck', 'trapezius', 'cat-cow']
        filtered_ex = [ex for ex in EXERCISES if not any(k in ex['name'].lower() for k in neck_keywords)]
        
        if state.exercise_index >= min(state.max_exercises, len(filtered_ex)):
            state.current_exercise = None
            return jsonify({"status": "READY_T2"})
            
        state.current_exercise = filtered_ex[state.exercise_index]
        # Clear metrics for the new exercise
        state.metrics['reps'] = 0
        state.metrics['stage'] = "Starting..."
        
        return jsonify({
            "status": "next", 
            "index": state.exercise_index,
            "exercise": state.current_exercise
        })
        
    elif action == "START_TEST2":
        state.phase = "TEST2"
        state.sub_phase = "IN_PROGRESS"
        analyzer.reset()
        return jsonify({"status": "started"})
        
    elif action == "END_TEST2":
        state.test2_score = analyzer.max_flex
        state.phase = "COMPLETE"
        state.sub_phase = "DONE"
        imp = state.test2_score - state.test1_score
        
        # Save to Supabase
        try:
            new_record = {
                "user_email": session.get('user'),
                "test1_score": state.test1_score,
                "test2_score": state.test2_score,
                "improvement": imp,
                "assessment_type": state.metrics.get('assessment_name', 'Toe-Touch'),
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            supabase.table('assessments').insert(new_record).execute()
        except Exception as e:
            print(f"Error saving assessment: {e}")

        return jsonify({
            "test1": state.test1_score,
            "test2": state.test2_score,
            "imp": imp
        })
        
    return jsonify({"error": "Invalid action"}), 400
        
@app.route('/chat', methods=['POST'])
def chat():
    """Improved AI Chatbot with Intent Understanding"""
    data = request.json
    msg = data.get('message', '').lower().strip()
    
    # Current Stats for Context
    score = state.metrics.get('flexibility_score', 0)
    phase = state.metrics.get('phase', 'IDLE')
    alignment = "optimal" if state.metrics.get('alignment_check') else "needing adjustment"
    
    # Intent-Response Database
    INTENTS = {
        "greeting": {
            "patterns": ["hi", "hello", "hey", "good morning", "greetings"],
            "responses": [
                f"Hello! I'm monitoring your {phase} phase. You're currently at {score}% flexibility. How can I assist?",
                "Hi there! Ready to work on your myofascial release? How are you feeling?",
                "Greetings! I'm your digital physiotherapist. What's on your mind today?"
            ]
        },
        "pain_query": {
            "patterns": ["pain", "hurt", "sharp", "ache", "sore", "stiff", "burning"],
            "responses": [
                "I'm sorry to hear you're in pain. Is it a sharp sensation or a dull ache? If it's sharp, please stop immediately. Since your score is " + str(score) + "%, we might need to slow down.",
                "Stiffness is common, but 'pain' is a signal to stop. Are you feeling this in your lower back or legs?",
                "Safety first: If the movement creates sharp joint pain, please exit the pose. A gentle stretch 'burn' is okay, but sharp pain is not."
            ]
        },
        "performance_query": {
            "patterns": ["how am i doing", "progress", "score", "flexibility", "good", "bad", "result"],
            "responses": [
                f"You're currently at {score}%. Based on your data, your alignment is {alignment}. Keep your knees locked for a better score!",
                f"Analysis shows a {score}% depth. To improve, focus on breathing out as you descend.",
                f"Your {phase} results show you're in the '{state.metrics.get('feedback', 'active')}' zone. You're doing well!"
            ]
        },
        "technique_query": {
            "patterns": ["how to", "technique", "form", "knees", "legs", "straight", "position", "camera"],
            "responses": [
                "For the most accurate analysis, stand exactly 6-8 feet away and stay sideways to the camera.",
                "Form Tip: Imagine a string pulling your head up before you bend. Keep those knees 'locked' but not strained.",
                "Your camera alignment is " + alignment + ". Try to ensure your whole body from head to toe is visible in the frame."
            ]
        },
        "encouragement": {
            "patterns": ["hard", "difficult", "tough", "can't do it", "struggling"],
            "responses": [
                "Every inch of progress counts! Flexibility takes time and consistent practice.",
                "Don't worry about the total depth today. Focus on the consistency of the movement.",
                "Your body will adapt. Just keep showing up for these daily assessments."
            ]
        }
    }

    import random
    
    # 1. Check for Pattern Matches
    found_intent = None
    for intent, data in INTENTS.items():
        if any(pattern in msg for pattern in data["patterns"]):
            found_intent = intent
            break
            
    # 2. Select Response
    if found_intent:
        response = random.choice(INTENTS[found_intent]["responses"])
    else:
        # 3. Fallback logic with a human-like touch
        if len(msg.split()) < 3:
            response = "I'm listening. Could you tell me a bit more about your physical state or how the exercise feels?"
        else:
            response = "That's a valid observation. Regarding your rehab, I would suggest focusing on slow, controlled movements. Would you like a tip on your current form?"

    return jsonify({"response": response})

@app.route('/reset', methods=['POST'])
def reset_assessment():
    state.reset()
    analyzer.reset()
    return jsonify({"status": "reset"})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Physio - Toe Touch Assessment System")
    print("="*50)
    print("Starting server at http://localhost:5000")
    print("Stand SIDEWAYS to camera for best results!")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
