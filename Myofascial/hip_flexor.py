import cv2
import numpy as np
import time
import math

class HipFlexorAnalyzer:
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_legacy = True
        except ImportError:
            print("MediaPipe not found. Please install it.")
        except AttributeError:
            import mediapipe as mp
            self.mp = mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request
            import os
            
            model_path = 'pose_landmarker_lite.task'
            if not os.path.exists(model_path):
                print("Downloading pose detection model...")
                url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
                urllib.request.urlretrieve(url, model_path)
                print("Model downloaded!")
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose = vision.PoseLandmarker.create_from_options(options)
            self.use_legacy = False
            self.frame_timestamp = 0

        # Colors (Premium Palette)
        self.colors = {
            'primary': (255, 149, 0),    # Orange
            'secondary': (0, 255, 127),  # Spring Green
            'background': (20, 20, 20),  # Dark Grey
            'text': (245, 245, 245),     # Off White
            'error': (59, 59, 255),      # Red
            'success': (80, 200, 120),   # Emerald
            'warning': (0, 165, 255),    # Gold
            'angle_arc': (255, 100, 100) # Light Red
        }
        
        # State tracking
        self.stage = "Setup" 
        self.feedback = "Kneel down. One knee on floor, other foot forward."
        self.hold_time = 0
        self.start_hold_time = None
        self.active_side = None
        
        self.current_metrics = {
            'hip_extension': 90, # Starts flexed/seated
            'torso_lean': 0,
            'front_knee_angle': 90,
            'form_status': 'Waiting'
        }

    def close(self):
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (p2 is vertex)."""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def calculate_vertical_angle(self, p1, p2):
        """Calculate angle of segment p1-p2 relative to vertical axis (0 is up/down)"""
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        vert = np.array([0, 1])
        v_norm = np.linalg.norm(v)
        if v_norm == 0: return 0
        v = v / v_norm
        dot = np.dot(v, vert)
        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        return angle

    def draw_premium_ui(self, frame):
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Dashboard Background
        cv2.rectangle(overlay, (20, 20), (380, 380), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Header
        cv2.putText(frame, "HIP FLEXOR ANALYSIS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (360, 65), self.colors['primary'], 2)
        
        # Timer Display
        timer_text = f"{self.hold_time:.1f}s"
        cv2.putText(frame, f"HOLD TIME: {timer_text}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['secondary'], 1)
        
        # Metrics Visualization
        # 1. Hip Extension (The key metric)
        # Should be > 170 (flat) or even > 180 (extended manually computed as > 180 if knee behind torso line)
        # Our angle calculation maxes at 180.
        # We want to be CLOSE to 180 to show "Open Hip".
        # If < 150 (sitting back), bad.
        hip_ext = self.current_metrics['hip_extension']
        
        # Map 130->180 to 0->100%
        ext_percent = max(0, min(100, (hip_ext - 130) / (180 - 130) * 100))
        
        bar_color = self.colors['warning']
        if hip_ext > 165: bar_color = self.colors['success']
        elif hip_ext < 140: bar_color = self.colors['error']
        
        cv2.putText(frame, f"Hip Openness: {int(hip_ext)}°", (40, 150), cv2.FONT_HERSHEY_DUPLEX, 0.6, bar_color, 1)
        cv2.rectangle(frame, (40, 160), (340, 175), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 160), (40 + int(3.0 * ext_percent), 175), bar_color, -1)
        
        # 2. Torso Upright
        # Should be small angle (< 10 deg)
        torso_angle = self.current_metrics['torso_lean']
        torso_color = self.colors['success'] if torso_angle < 15 else self.colors['warning']
        if torso_angle > 25: torso_color = self.colors['error']
        
        cv2.putText(frame, f"Torso Lean: {int(torso_angle)}°", (40, 215), cv2.FONT_HERSHEY_DUPLEX, 0.6, torso_color, 1)
        
        # 3. Side
        active_side_text = self.active_side.title() if self.active_side else "None"
        cv2.putText(frame, f"Checking Side: {active_side_text}", (40, 255), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)

        # Form Status
        status_color = self.colors['success'] if "Good" in self.current_metrics['form_status'] else self.colors['warning']
        if "Bad" in self.current_metrics['form_status']: status_color = self.colors['error']
        
        cv2.putText(frame, f"Status: {self.current_metrics['form_status']}", (40, 320), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1)

        # Feedback Box
        cv2.rectangle(overlay, (20, h-80), (w-20, h-20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"FEEDBACK: {self.feedback}", (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

    def draw_skeleton(self, frame, landmarks, w, h):
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                cv2.line(frame, (int(start.x * w), int(start.y * h)), (int(end.x * w), int(end.y * h)), self.colors['primary'], 2)
        
        # Highlight Knees
        for idx in [25, 26]:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 6, self.colors['secondary'], -1)

    def analyze(self, frame):
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        processed_landmarks = None
        
        if self.use_legacy:
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                processed_landmarks = results.pose_landmarks.landmark
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            self.frame_timestamp += 1
            results = self.pose.detect_for_video(mp_image, self.frame_timestamp)
            if results.pose_landmarks:
                processed_landmarks = results.pose_landmarks[0]

        if processed_landmarks:
            self.draw_skeleton(frame, processed_landmarks, w, h)
            
            def get_pt(idx):
                return (processed_landmarks[idx].x * w, processed_landmarks[idx].y * h)

            # Metrics
            l_sh = get_pt(11)
            r_sh = get_pt(12)
            l_hip = get_pt(23)
            r_hip = get_pt(24)
            l_knee = get_pt(25)
            r_knee = get_pt(26)
            l_ankle = get_pt(27)
            r_ankle = get_pt(28)
            
            # Detect Kneeling State: Which knee is lower (closer to ground)?
            # In image coords, larger Y = lower
            l_knee_y = l_knee[1]
            r_knee_y = r_knee[1]
            
            # Determine threshold for "on floor" relative to hip height?
            # Or just check relative position.
            
            # Usually strict kneeling: one knee significantly lower than other?
            # Or one foot forward and one back.
            
            is_kneeling = False
            active_leg = None # The rear leg being stretched
            
            # Heuristic: Rear leg knee is much lower (larger Y) than Front leg knee?
            # No, front leg knee is UP (smaller Y). Rear leg knee is DOWN (larger Y).
            
            # Check difference
            diff_y = abs(l_knee_y - r_knee_y)
            # Needs to be significant, e.g., > 10% of height?
            threshold = h * 0.1
            
            if diff_y > threshold:
                is_kneeling = True
                if l_knee_y > r_knee_y:
                    active_leg = 'left' # Left knee is down (rear)
                else:
                    active_leg = 'right' # Right knee is down (rear)
            
            self.active_side = active_leg
            
            curr_hip_ext = 0
            curr_torso = 0
            
            if is_kneeling and active_leg:
                if active_leg == 'left':
                    # Calculate Left Hip Extension (Shoulder-Hip-Knee)
                    # We want this to be OPEN (~180).
                    curr_hip_ext = self.calculate_angle(l_sh, l_hip, l_knee)
                    
                    # Torso Lean (Shoulder relative to Hip vertical)
                    curr_torso = self.calculate_vertical_angle(l_hip, l_sh)
                    
                else:
                    curr_hip_ext = self.calculate_angle(r_sh, r_hip, r_knee)
                    curr_torso = self.calculate_vertical_angle(r_hip, r_sh)
                    
                self.current_metrics['hip_extension'] = curr_hip_ext
                self.current_metrics['torso_lean'] = curr_torso
                
                # Check Form
                is_good_form = True
                
                # 1. Torso Upright?
                if curr_torso > 20: 
                    self.feedback = "Lean back! Keep torso upright."
                    is_good_form = False
                    self.current_metrics['form_status'] = 'Lean Back'
                
                # 2. Hip Open?
                elif curr_hip_ext < 155:
                    self.feedback = "Push hips forward to open stretch."
                    is_good_form = False
                    self.current_metrics['form_status'] = 'Push Hips'
                
                else:
                    self.feedback = "Great stretch! Hold it."
                    self.current_metrics['form_status'] = 'Good Stretch'
                    
                # Timer Logic
                if is_good_form:
                    if self.start_hold_time is None:
                        self.start_hold_time = time.time()
                    else:
                        self.hold_time = time.time() - self.start_hold_time
                else:
                    self.start_hold_time = None 
                    
            else:
                self.feedback = "Kneel with one knee down, one foot forward."
                self.current_metrics['form_status'] = 'Waiting'
                self.current_metrics['hip_extension'] = 90
                self.start_hold_time = None

        # self.draw_premium_ui(frame)
        return frame

    def save_report(self, frame):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"hip_flexor_analysis_{timestamp}.png"
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "HIP FLEXOR STRETCH REPORT", (w // 2 - 250, h + 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        cv2.putText(report, f"Max Hold Time: {self.hold_time:.1f}s", (50, h + 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.imwrite(filename, report)
        print(f"Report saved: {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = HipFlexorAnalyzer()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Premium Hip Flexor Analysis', cv2.WINDOW_NORMAL)
    
    print("--- Hip Flexor Stretch Analysis Started ---")
    print("Press 'q' to quit, 's' to save report")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Premium Hip Flexor Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Quit command received.")
            break
        elif key == ord('s') or key == ord('S'):
            analyzer.save_report(processed_frame)
            
        if cv2.getWindowProperty('Premium Hip Flexor Analysis', cv2.WND_PROP_VISIBLE) < 1:
             print("Window closed.")
             break

    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
