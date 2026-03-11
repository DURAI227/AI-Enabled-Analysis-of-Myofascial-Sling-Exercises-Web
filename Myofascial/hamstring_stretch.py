import cv2
import numpy as np
import time
import math

class HamstringStretchAnalyzer:
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
        self.stage = "Relaxed" 
        self.feedback = "Stand sideways. Place foot forward or on a surface."
        self.hold_time = 0
        self.start_hold_time = None
        self.max_stretch_angle = 180 # Hip angle starts at 180 (straight)
        self.current_metrics = {
            'knee_angle': 0,
            'hip_flexion': 180,
            'torso_angle': 0,
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

    def draw_premium_ui(self, frame):
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Dashboard Background
        cv2.rectangle(overlay, (20, 20), (380, 360), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Header
        cv2.putText(frame, "HAMSTRING STRETCH ANALYSIS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (360, 65), self.colors['primary'], 2)
        
        # Timer Display
        timer_text = f"{self.hold_time:.1f}s"
        cv2.putText(frame, f"HOLD TIME: {timer_text}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['secondary'], 1)
        
        # Metrics Visualization
        # 1. Knee Extension (Straightness)
        knee_angle = self.current_metrics['knee_angle']
        knee_color = self.colors['success'] if knee_angle > 165 else self.colors['warning']
        if knee_angle < 140: knee_color = self.colors['error']
        
        cv2.putText(frame, f"Knee Straightness: {int(knee_angle)}°", (40, 150), cv2.FONT_HERSHEY_DUPLEX, 0.6, knee_color, 1)
        
        # 2. Hip Flexion (Stretch Depth)
        hip_angle = self.current_metrics['hip_flexion']
        # The lower the better (bending forward)
        # Typically start ~170, good stretch ~120-90
        stretch_percent = max(0, min(100, (170 - hip_angle) / (170 - 90) * 100))
        
        cv2.putText(frame, f"Stretch Depth: {int(stretch_percent)}%", (40, 190), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.rectangle(frame, (40, 200), (340, 215), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 200), (40 + int(3.0 * stretch_percent), 215), self.colors['primary'], -1)
        
        # 3. Torso Check (optional based on rounded back, simplistic check here)
        torso_angle = self.current_metrics['torso_angle']
        cv2.putText(frame, f"Torso Angle: {int(torso_angle)}°", (40, 245), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)

        # Form Status
        status_color = self.colors['success'] if self.current_metrics['form_status'] == 'Good Stretch' else self.colors['warning']
        if "Bad" in self.current_metrics['form_status']: status_color = self.colors['error']
        
        cv2.putText(frame, f"Status: {self.current_metrics['form_status']}", (40, 310), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1)

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
        
        # Highlight Knees and Hips
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
            
            # --- Metrics Calculation ---
            def get_pt(idx):
                return (processed_landmarks[idx].x * w, processed_landmarks[idx].y * h)

            # Left Side metrics
            l_sh = get_pt(11)
            l_hip = get_pt(23)
            l_knee = get_pt(25)
            l_ankle = get_pt(27)
            
            # Right Side metrics
            r_sh = get_pt(12)
            r_hip = get_pt(24)
            r_knee = get_pt(26)
            r_ankle = get_pt(28)
            
            # Determine which side is facing camera / active
            # Simple heuristic: Side with more visible hip flexion (smaller angle)
            # Or assume user is sideways and pick leg closer to camera (using Z if available, but let's use 2D visibility/angle)
            
            l_hip_angle = self.calculate_angle(l_sh, l_hip, l_knee)
            r_hip_angle = self.calculate_angle(r_sh, r_hip, r_knee)
            
            # Use the "more bent" hip as the active side (stretching side usually has straight knee, but bent hip relative to torso)
            # Actually, both legs might hinge.
            # Let's filter by Knee Extension. The STRETCHING leg MUST be straight.
            l_knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
            
            active_side = None
            
            # Check for straight knee (> 160 deg)
            l_straight = l_knee_angle > 155
            r_straight = r_knee_angle > 155
            
            if l_straight and r_straight:
                # Both straight: pick the one with smaller hip angle (forward bend)
                if l_hip_angle < r_hip_angle: active_side = 'left'
                else: active_side = 'right'
            elif l_straight:
                active_side = 'left'
            elif r_straight:
                active_side = 'right'
            else:
                # Neither straight
                active_side = 'none'
            
            if active_side == 'left':
                curr_knee = l_knee_angle
                curr_hip = l_hip_angle
                # Torso angle relative to vertical? Or relative to leg?
                # Hip angle already captures torso-leg relation.
                curr_torso = self.calculate_angle((l_sh[0], 0), l_sh, l_hip) # Approx vertical
            elif active_side == 'right':
                curr_knee = r_knee_angle
                curr_hip = r_hip_angle
                curr_torso = self.calculate_angle((r_sh[0], 0), r_sh, r_hip)
            else:
                curr_knee = (l_knee_angle + r_knee_angle) / 2
                curr_hip = (l_hip_angle + r_hip_angle) / 2
                curr_torso = 0

            self.current_metrics['knee_angle'] = curr_knee
            self.current_metrics['hip_flexion'] = curr_hip
            self.current_metrics['torso_angle'] = curr_torso
            
            # --- Logic Flow ---
            if active_side != 'none':
                # Valid stretching posture candidate
                if curr_knee > 165:
                    if curr_hip < 160: # Hinging forward
                        # Start Timer
                        if self.start_hold_time is None:
                            self.start_hold_time = time.time()
                            self.feedback = "Good stretch! Hold it."
                        else:
                            self.hold_time = time.time() - self.start_hold_time
                            # Encourage depth
                            if curr_hip < 120:
                                self.feedback = "Deep stretch! Breathe."
                            else:
                                self.feedback = "Lean forward from hips to deepen."
                        self.current_metrics['form_status'] = 'Good Stretch'
                    else:
                        # Standing straight with straight leg
                        self.start_hold_time = None
                        self.hold_time = 0
                        self.feedback = "Lean forward to feel the stretch."
                        self.current_metrics['form_status'] = 'Waiting'
                else:
                    # Knee bent
                    self.start_hold_time = None 
                    self.feedback = "Straighten your knee!"
                    self.current_metrics['form_status'] = 'Bad Form'
            else:
                self.start_hold_time = None
                self.feedback = "Straighten at least one leg."
                self.current_metrics['form_status'] = 'Waiting'

        # self.draw_premium_ui(frame)
        return frame

    def save_report(self, frame):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"hamstring_stretch_analysis_{timestamp}.png"
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "HAMSTRING STRETCH REPORT", (w // 2 - 250, h + 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        cv2.putText(report, f"Max Hold Time: {self.hold_time:.1f}s", (50, h + 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.imwrite(filename, report)
        print(f"Report saved: {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = HamstringStretchAnalyzer()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Premium Hamstring Stretch Analysis', cv2.WINDOW_NORMAL)
    
    print("--- Hamstring Stretch Analysis Started ---")
    print("Press 'q' to quit, 's' to save report")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Premium Hamstring Stretch Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Quit command received.")
            break
        elif key == ord('s') or key == ord('S'):
            analyzer.save_report(processed_frame)
            
        if cv2.getWindowProperty('Premium Hamstring Stretch Analysis', cv2.WND_PROP_VISIBLE) < 1:
             print("Window closed.")
             break

    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
