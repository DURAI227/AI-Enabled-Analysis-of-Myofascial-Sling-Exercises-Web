import cv2
import numpy as np
import math
import time

class ForwardLungeAnalyzer:
    def __init__(self):
        # Import mediapipe here to handle version differences and avoid global naming conflicts
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_legacy = True
        except ImportError:
             print("MediaPipe not found. Please install it.")
        except AttributeError:
            # Use new MediaPipe Tasks API if legacy not available or preferred
            import mediapipe as mp
            self.mp = mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download the pose landmarker model if not exists
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
            'warning': (0, 165, 255)     # Gold
        }
        
        # State tracking
        self.stage = "Standing"  # Standing, Lunging Left (Left Forward), Lunging Right (Right Forward)
        self.feedback = "Stand with feet hip-width. Step forward to lunge."
        self.rep_count_left = 0
        self.rep_count_right = 0
        self.current_side = None # 'left' or 'right' indicating FRONT leg
        self.current_metrics = {
            'front_knee_angle': 0,
            'back_knee_angle': 0,
            'torso_angle': 0,
            'depth_percent': 0,
            'form_status': 'Good'
        }

    def close(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'pose') and self.pose:
            if self.use_legacy:
                self.pose.close()
            else:
                self.pose.close()

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (p2 is the vertex)"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def calculate_vertical_angle(self, p1, p2):
        """Calculate angle of segment p1-p2 relative to vertical axis"""
        # Vector p1->p2
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        # Vertical vector (pointing down usually in image coords)
        vert = np.array([0, 1])
        
        # Normalize
        v_norm = np.linalg.norm(v)
        if v_norm == 0: return 0
        v = v / v_norm
        
        dot = np.dot(v, vert)
        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        return angle

    def draw_premium_ui(self, frame):
        h, w, _ = frame.shape
        # Create a semi-transparent overlay for the dashboard
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (380, 320), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status Bar
        status_color = self.colors['success'] if "Perfect" in self.feedback or "Great" in self.feedback else self.colors['primary']
        if "Alert" in self.feedback or "Warning" in self.feedback:
            status_color = self.colors['error']
        elif "Keep" in self.feedback:
            status_color = self.colors['warning']
            
        cv2.putText(frame, "FORWARD LUNGE ANALYSIS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (360, 65), self.colors['primary'], 2)
        
        cv2.putText(frame, f"L FRONT: {self.rep_count_left} | R FRONT: {self.rep_count_right}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        cv2.putText(frame, f"STAGE: {self.stage}", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        
        # Metrics Visualization
        # Front Knee Angle
        cv2.putText(frame, f"Front Knee: {int(self.current_metrics['front_knee_angle'])}°", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        # Torso Angle (Visual check for upright)
        torso_color = self.colors['success'] if self.current_metrics['torso_angle'] < 15 else self.colors['warning']
        cv2.putText(frame, f"Torso Lean: {int(self.current_metrics['torso_angle'])}°", (40, 200), cv2.FONT_HERSHEY_DUPLEX, 0.6, torso_color, 1)

        # Depth Progress Bar
        depth_progress = min(100, self.current_metrics['depth_percent'])
        cv2.putText(frame, f"Depth: {int(depth_progress)}%", (40, 230), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.rectangle(frame, (40, 240), (340, 255), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 240), (40 + int(3.0 * depth_progress), 255), status_color, -1)
        
        # Form Status
        cv2.putText(frame, f"Form: {self.current_metrics['form_status']}", (40, 285), cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 1)

        # Feedback Box at bottom
        cv2.rectangle(overlay, (20, h-80), (w-20, h-20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"FEEDBACK: {self.feedback}", (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

    def draw_skeleton(self, frame, landmarks, w, h):
        """Draw skeleton overlay on frame"""
        # Define connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27), (27, 29), (29, 31),  # Left Leg
            (24, 26), (26, 28), (28, 30), (30, 32)   # Right Leg
        ]
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, self.colors['primary'], 2)
        
        # Draw landmarks
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

    def analyze(self, frame):
        h, w, _ = frame.shape
        
        if self.use_legacy:
            # Legacy MediaPipe API
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Draw skeleton
                self.mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                    self.mp_draw.DrawingSpec(color=self.colors['primary'], thickness=2, circle_radius=2)
                )
                
                # Process landmarks
                self.process_landmarks(landmarks, w, h)
        else:
            # New MediaPipe Tasks API
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            
            self.frame_timestamp += 1
            results = self.pose.detect_for_video(mp_image, self.frame_timestamp)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                
                # Draw skeleton
                self.draw_skeleton(frame, landmarks, w, h)
                
                # Process landmarks
                self.process_landmarks(landmarks, w, h)
        
        # self.draw_premium_ui(frame)
        return frame

    def process_landmarks(self, landmarks, w, h):
        """Process pose landmarks for forward lunge analysis"""
        def get_coords(idx):
            lm = landmarks[idx]
            return (int(lm.x * w), int(lm.y * h))
        
        # Landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        
        # Get coordinates
        l_hip = get_coords(LEFT_HIP)
        r_hip = get_coords(RIGHT_HIP)
        l_knee = get_coords(LEFT_KNEE)
        r_knee = get_coords(RIGHT_KNEE)
        l_ankle = get_coords(LEFT_ANKLE)
        r_ankle = get_coords(RIGHT_ANKLE)
        l_sh = get_coords(LEFT_SHOULDER)
        r_sh = get_coords(RIGHT_SHOULDER)
        
        # Calculate Knee Angles
        left_knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
        
        # Determine Lunge State & Front Leg
        # In a deep lunge, both knees are bent, usually < 150 deg (allowing for transition)
        is_lunging = left_knee_angle < 150 and right_knee_angle < 150
        
        # Determine "Front" leg by checking which shin is more vertical
        # The front leg's shin (Ankle to Knee) is usually vertical in a proper lunge
        l_shin_vert = self.calculate_vertical_angle(l_knee, l_ankle)
        r_shin_vert = self.calculate_vertical_angle(r_knee, r_ankle)
        
        # Initial check for "Front Leg" based on verticality of shin
        # The leg with the smaller vertical angle is typically the front leg
        if l_shin_vert < r_shin_vert:
            front_leg = "left"
            front_knee_angle = left_knee_angle
            back_knee_angle = right_knee_angle
            front_shin_vert = l_shin_vert
            
            torso_mid = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
            hip_mid = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
            torso_angle = self.calculate_vertical_angle(hip_mid, torso_mid)
        else:
            front_leg = "right"
            front_knee_angle = right_knee_angle
            back_knee_angle = left_knee_angle
            front_shin_vert = r_shin_vert
            
            torso_mid = ((l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2)
            hip_mid = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
            torso_angle = self.calculate_vertical_angle(hip_mid, torso_mid)

        # Depth Metrics
        # 90 degrees is ideal ~ 100%
        # 180 is standing ~ 0%
        depth_percent = max(0, min(100, (180 - front_knee_angle) / 90 * 100))
        
        self.current_metrics['front_knee_angle'] = front_knee_angle
        self.current_metrics['back_knee_angle'] = back_knee_angle
        self.current_metrics['torso_angle'] = torso_angle
        self.current_metrics['depth_percent'] = depth_percent

        # Logic Flow
        if is_lunging:
            # Update Stage
            if self.stage == "Standing":
                self.stage = f"Lunging {front_leg.title()}"
                self.current_side = front_leg
            
            # Feedback
            self.current_metrics['form_status'] = 'Good'
            
            if abs(torso_angle) > 20:
                self.feedback = "Keep your torso upright!"
                self.current_metrics['form_status'] = 'Poor'
            elif front_shin_vert > 25: 
                # Slightly more lenient here as forward lunge often has some forward knee travel
                self.feedback = "Ensure knee doesn't drift too far past toes."
                self.current_metrics['form_status'] = 'Warning'
            elif front_knee_angle < 75:
                 self.feedback = "Too deep! Dont bang your back knee."
            elif front_knee_angle > 115:
                self.feedback = "Go lower! Aim for 90 degrees."
            elif 80 <= front_knee_angle <= 100:
                self.feedback = "Perfect depth! Push back to start."
                self.current_metrics['form_status'] = 'Perfect'
        else:
            # Standing or transition
            if "Lunging" in self.stage:
                # We were lunging, now we are standing -> Count Rep
                # Validate the return to neutral (standing)
                if self.stage == "Lunging Left" and left_knee_angle > 160: 
                    self.rep_count_left += 1
                    self.feedback = "Good left rep! Switch legs."
                    self.stage = "Standing"
                    self.current_side = None
                elif self.stage == "Lunging Right" and right_knee_angle > 160:
                    self.rep_count_right += 1
                    self.feedback = "Good right rep! Switch legs."
                    self.stage = "Standing"
                    self.current_side = None
            else:
                 self.feedback = "Stand tall. Step forward into lunge."
                 self.current_metrics['form_status'] = 'Waiting'

    def save_report(self, frame):
        """Saves a 'Post Analysis' summary of the session."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"forward_lunge_analysis_{timestamp}.png"
        
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "FORWARD LUNGE POSTURE ANALYSIS REPORT", (w // 2 - 280, h + 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        
        cv2.putText(report, f"Left Front Reps: {self.rep_count_left} | Right Front Reps: {self.rep_count_right}", (50, h + 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        cv2.putText(report, f"Last Status: {self.current_metrics['form_status']}", 
                    (50, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        cv2.imwrite(filename, report)
        print(f"Analysis saved as {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = ForwardLungeAnalyzer()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("--- Forward Lunge Analysis Started ---")
    print("Press 's' to save a Report Poster")
    print("Press 'q' to quit")

    cv2.namedWindow('Premium Forward Lunge Analysis', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Premium Forward Lunge Analysis', processed_frame)
        
        # Check for key press (wait 1ms)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27: # q or Q or ESC
            print("Quit command received.")
            break
        elif key == ord('s') or key == ord('S'):
            analyzer.save_report(processed_frame)
            
        # Additional check in case window is closed manually
        if cv2.getWindowProperty('Premium Forward Lunge Analysis', cv2.WND_PROP_VISIBLE) < 1:
             print("Window closed.")
             break

    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
