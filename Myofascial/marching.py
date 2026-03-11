import cv2
import numpy as np
import time
import math

class MarchingAnalyzer:
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
        self.feedback = "Stand tall. March knees high."
        self.rep_count = 0
        self.current_side = None # 'Left' or 'Right'
        
        self.current_metrics = {
            'knee_height_score': 0, # 0-100%
            'torso_lean': 0,
            'balance_score': 100,
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
        cv2.putText(frame, "MARCHING ANALYSIS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (360, 65), self.colors['primary'], 2)
        
        # Stats
        cv2.putText(frame, f"REPS: {self.rep_count}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        side_text = f"Lift: {self.current_side}" if self.current_side else "Ready"
        cv2.putText(frame, side_text, (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['secondary'], 1)
        
        # Metrics Visualization
        # 1. Knee Lift Height (Hip Flexion)
        lift_score = self.current_metrics['knee_height_score']
        # 0% is standing, 100% is thigh horizontal (or > 90 deg hip flexion)
        
        bar_color = self.colors['warning']
        if lift_score > 80: bar_color = self.colors['success']
        if lift_score < 20: bar_color = self.colors['error']
        
        cv2.putText(frame, f"Knee Lift: {int(lift_score)}%", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.rectangle(frame, (40, 180), (340, 195), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 180), (40 + int(3.0 * lift_score), 195), bar_color, -1)
        
        # 2. Torso Stability
        lean = self.current_metrics['torso_lean']
        lean_color = self.colors['success'] if lean < 10 else self.colors['error']
        cv2.putText(frame, f"Torso Lean: {int(lean)}°", (40, 230), cv2.FONT_HERSHEY_DUPLEX, 0.6, lean_color, 1)

        # Form Status
        status_color = self.colors['success'] if "Good" in self.current_metrics['form_status'] else self.colors['warning']
        if "Bad" in self.current_metrics['form_status'] or "Lean" in self.current_metrics['form_status']: 
            status_color = self.colors['error']
        
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
            mid_hip = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
            mid_sh = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
            
            # 1. Determine which leg is raised
            # Compare knee height (Y coord, smaller is higher) relative to hip
            l_knee_height = l_hip[1] - l_knee[1]
            r_knee_height = r_hip[1] - r_knee[1]
            
            # Threshold for "Active Lift": Knee needs to be significantly closer to hip level
            # If distance is small, leg is down (knee ~2 units below hip). If high, knee ~0 units below hip.
            # Usually knee is ~0.4 * height below hip when standing.
            # When marching, knee Y approaches Hip Y.
            
            # Angles are better.
            l_hip_angle = self.calculate_angle(l_sh, l_hip, l_knee)
            r_hip_angle = self.calculate_angle(r_sh, r_hip, r_knee)
            
            active_angle = 180 # default
            
            # Check flexion (< 150 start to count as move)
            if l_hip_angle < 160 and r_hip_angle > 160:
                side = 'Left'
                active_angle = l_hip_angle
            elif r_hip_angle < 160 and l_hip_angle > 160:
                side = 'Right'
                active_angle = r_hip_angle
            elif l_hip_angle < 160 and r_hip_angle < 160:
                # Both bent? Squat/Sit? Or jumping?
                side = 'Both?'
                active_angle = min(l_hip_angle, r_hip_angle)
            else:
                side = None
            
            # Calculate Torso Lean
            torso_lean = self.calculate_vertical_angle(mid_hip, mid_sh)
            self.current_metrics['torso_lean'] = torso_lean
            
            # Calculate Lift Score (Goal: 90 deg hip angle or less)
            # Map 170 (Standing) -> 90 (High Knee) to 0 -> 100%
            lift_score = max(0, min(100, (170 - active_angle) / (170 - 90) * 100))
            self.current_metrics['knee_height_score'] = lift_score
            self.current_side = side

            # State Machine for Reps
            if side:
                if torso_lean > 20:
                     self.feedback = "Keep torso upright!"
                     self.current_metrics['form_status'] = 'Lean Back'
                elif lift_score > 80: # Great height
                     self.feedback = f"Great {side} Knee High!"
                     self.current_metrics['form_status'] = 'Good Rep'
                     if self.stage == "Standing":
                         self.stage = f"Lift {side}"
                elif lift_score > 40:
                     self.feedback = f"Drive that {side} knee higher!"
                     self.current_metrics['form_status'] = 'Driving'
                     if self.stage == "Standing":
                         self.stage = f"Lift {side}"
            else:
                # Standing or returning
                if "Lift" in self.stage:
                    # Just finished a rep
                    self.rep_count += 1
                    self.stage = "Standing"
                    self.feedback = "Good switch. Next leg!"
                else:
                    self.feedback = "March in place. Knees up!"
                    self.current_metrics['form_status'] = 'Waiting'

        # self.draw_premium_ui(frame)
        return frame

    def save_report(self, frame):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"marching_analysis_{timestamp}.png"
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "MARCHING EXERCISE REPORT", (w // 2 - 250, h + 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        cv2.putText(report, f"Total Steps: {self.rep_count}", (50, h + 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.imwrite(filename, report)
        print(f"Report saved: {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = MarchingAnalyzer()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Premium Marching Analysis', cv2.WINDOW_NORMAL)
    
    print("--- Marching Analysis Started ---")
    print("Press 'q' to quit, 's' to save report")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Premium Marching Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Quit command received.")
            break
        elif key == ord('s') or key == ord('S'):
            analyzer.save_report(processed_frame)
            
        if cv2.getWindowProperty('Premium Marching Analysis', cv2.WND_PROP_VISIBLE) < 1:
             print("Window closed.")
             break

    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
