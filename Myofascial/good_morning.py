import cv2
import numpy as np
import time
import math

class GoodMorningAnalyzer:
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
        }
        
        # State tracking
        self.stage = "Standing"
        self.feedback = "Stand sideways. Hinge at hips, keep back straight."
        self.rep_count = 0
        self.min_hip_angle = 180
        
        self.current_metrics = {
            'hip_angle': 180,
            'knee_angle': 180,
            'torso_inclination': 0,
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
        
    def calculate_inclination(self, p1, p2):
        """Calculate angle relative to vertical (0 is up)."""
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        vert = np.array([0, -1]) # Up vector (y decreases going up)
        # Actually in screen coords locally:
        # Standard: (0,0) top left. 
        # Vector p1->p2 (Hip->Shoulder) should be UP.
        # Hip is lower (larger Y), Shoulder higher (smaller Y).
        # v = (sx - hx, sy - hy). sy < hy, so sy-hy is negative.
        # This aligns with (0, -1).
        
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
        cv2.putText(frame, "GOOD MORNING ANALYSIS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (360, 65), self.colors['primary'], 2)
        
        # Stats
        cv2.putText(frame, f"REPS: {self.rep_count}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        cv2.putText(frame, f"STAGE: {self.stage}", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['secondary'], 1)
        
        # Metrics Visualization
        # 1. Hip Angle (The Hinge)
        hip_angle = self.current_metrics['hip_angle']
        # 180 is straight, ~90 is bottom
        
        # Color coding
        hinge_color = self.colors['warning']
        if 80 <= hip_angle <= 110: hinge_color = self.colors['success'] # Good depth
        elif hip_angle < 70: hinge_color = self.colors['error'] # Too deep/collapsed
        
        cv2.putText(frame, f"Hip Angle: {int(hip_angle)}°", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        # Bar: 180 -> 100% full, 90 -> 50%? No, show depth progress?
        # Let's map 180 (start) to 0% progress, 90 (end) to 100% progress.
        progress = max(0, min(100, (180 - hip_angle) / (180 - 90) * 100))
        
        cv2.rectangle(frame, (40, 180), (340, 195), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 180), (40 + int(3.0 * progress), 195), hinge_color, -1)
        
        # 2. Knee Bend Check (Should be slight)
        knee_angle = self.current_metrics['knee_angle']
        k_color = self.colors['success'] if knee_angle > 150 else self.colors['error']
        cv2.putText(frame, f"Knee Angle: {int(knee_angle)}°", (40, 230), cv2.FONT_HERSHEY_DUPLEX, 0.6, k_color, 1)

        # Form Status
        status_color = self.colors['success'] if "Good" in self.current_metrics['form_status'] else self.colors['warning']
        if "Bad" in self.current_metrics['form_status'] or "Squat" in self.current_metrics['form_status']:
            status_color = self.colors['error']
        
        cv2.putText(frame, f"Status: {self.current_metrics['form_status']}", (40, 320), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1)

        # Feedback Box
        cv2.rectangle(overlay, (20, h-80), (w-20, h-20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"FEEDBACK: {self.feedback}", (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

    def draw_skeleton(self, frame, landmarks, w, h):
        connections = [
            (11, 12), (11, 13), (12, 14), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                cv2.line(frame, (int(start.x * w), int(start.y * h)), (int(end.x * w), int(end.y * h)), self.colors['primary'], 2)
        
        # Highlight Hips and Knees
        for idx in [23, 24, 25, 26]:
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

            # Determine side (Left/Right)
            # Use visibility or Z-depth. For simplicity, check which shoulder is more visible/confident if available.
            # Or usually side view: check variance in x coords?
            # Standard: Use side with more prominent landmarks.
            
            # Let's compute both side angles and take the one that looks more "active" (hinging)?
            l_hip_angle = self.calculate_angle(get_pt(11), get_pt(23), get_pt(25)) # L_Sh, L_Hip, L_Knee
            r_hip_angle = self.calculate_angle(get_pt(12), get_pt(24), get_pt(26))
            
            # Use the smaller angle (more bent) as the indicator of the hinge
            if l_hip_angle < r_hip_angle:
                side = 'left'
                hip_angle = l_hip_angle
                knee_angle = self.calculate_angle(get_pt(23), get_pt(25), get_pt(27)) # Hip, Knee, Ankle
                torso_pts = (get_pt(23), get_pt(11)) # Hip, Shoulder
            else:
                side = 'right'
                hip_angle = r_hip_angle
                knee_angle = self.calculate_angle(get_pt(24), get_pt(26), get_pt(28))
                torso_pts = (get_pt(24), get_pt(12))

            # Inclination (0 is standing upright)
            inclination = self.calculate_inclination(torso_pts[0], torso_pts[1])
            
            self.current_metrics['hip_angle'] = hip_angle
            self.current_metrics['knee_angle'] = knee_angle
            self.current_metrics['torso_inclination'] = inclination
            
            # --- Logic Flow ---
            
            # Form Check
            is_good_form = True
            
            # 1. Knee Check (Should be "soft", 150-170, not < 140)
            if knee_angle < 140:
                self.feedback = "Don't Squat! Keep legs straighter."
                self.current_metrics['form_status'] = 'Squatting'
                is_good_form = False
            elif knee_angle > 175:
                # Locked knees? Acceptable but "soft bend" is better.
                pass
            
            # 2. Hinge Check
            # State Machine
            
            # Thresholds
            START_THRESH = 160 # Standing
            BOTTOM_THRESH = 110 # Hinge point (approx 90-110 deg)
            
            if self.stage == "Standing":
                if hip_angle < 150: # Started hinging
                    self.stage = "Hinging"
                    self.feedback = "Push hips back..."
                    self.min_hip_angle = hip_angle
            
            elif self.stage == "Hinging":
                # Track depth
                if hip_angle < self.min_hip_angle:
                    self.min_hip_angle = hip_angle
                
                if hip_angle < BOTTOM_THRESH:
                    if is_good_form:
                        self.stage = "Bottom"
                        self.feedback = "Good depth! Drive hips forward."
                    else:
                        self.feedback = "Watch your knees!"
                
                # Use inflection point? If angle starts increasing significantly?
                if hip_angle > self.min_hip_angle + 10 and self.min_hip_angle > BOTTOM_THRESH:
                     # Returned early
                     self.stage = "Standing"
                     self.feedback = "Go deeper next time."

            elif self.stage == "Bottom":
                if hip_angle > 140: # Returning up
                    if is_good_form:
                        self.rep_count += 1
                        self.feedback = "Good Rep!"
                        self.current_metrics['form_status'] = 'Good Rep'
                    self.stage = "Standing"
                else:
                    self.feedback = "Squeeze glutes to stand up."
            
            # General feedback if form is good but stage feedback not specific
            if is_good_form and "Rep" not in self.stage and "Bottom" not in self.stage:
                if inclination > 20: 
                    self.current_metrics['form_status'] = 'Hinging'
                else:
                     self.current_metrics['form_status'] = 'Standing'

        # self.draw_premium_ui(frame)
        return frame

    def save_report(self, frame):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"good_morning_analysis_{timestamp}.png"
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "GOOD MORNING EXERCISE REPORT", (w // 2 - 250, h + 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        cv2.putText(report, f"Total Reps: {self.rep_count}", (50, h + 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.imwrite(filename, report)
        print(f"Report saved: {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = GoodMorningAnalyzer()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Premium Good Morning Analysis', cv2.WINDOW_NORMAL)
    
    print("--- Good Morning Analysis Started ---")
    print("Press 'q' to quit, 's' to save report")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Premium Good Morning Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Quit command received.")
            break
        elif key == ord('s') or key == ord('S'):
            analyzer.save_report(processed_frame)
            
        if cv2.getWindowProperty('Premium Good Morning Analysis', cv2.WND_PROP_VISIBLE) < 1:
             print("Window closed.")
             break

    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
