import cv2
import numpy as np
import math
import time

class LateralLungeAnalyzer:
    def __init__(self):
        # Import mediapipe here to handle version differences
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
        except AttributeError:
            # Use new MediaPipe Tasks API
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download the pose landmarker model if not exists
            import urllib.request
            import os
            
            self.mp = mp
            
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
        self.stage = "Standing"  # Standing, Lunging Left, Lunging Right
        self.feedback = "Stand with feet hip-width apart. Step to the side."
        self.rep_count_left = 0
        self.rep_count_right = 0
        self.current_side = None
        self.lunge_depth = 0
        self.current_metrics = {
            'knee_angle': 0,
            'knee_alignment': 'Good',
            'torso_angle': 0,
            'depth_percent': 0
        }

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

    def get_line_angle(self, p1, p2):
        """Calculates the angle of the line p1-p2 with respect to the horizontal."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0: angle += 360
        return angle

    def draw_premium_ui(self, frame):
        h, w, _ = frame.shape
        # Create a semi-transparent overlay for the dashboard
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (380, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status Bar
        status_color = self.colors['success'] if "Perfect" in self.feedback or "Great" in self.feedback else self.colors['primary']
        if "Alert" in self.feedback or "Warning" in self.feedback:
            status_color = self.colors['error']
        elif "Keep" in self.feedback:
            status_color = self.colors['warning']
            
        cv2.putText(frame, "LATERAL LUNGE ANALYSIS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (360, 65), self.colors['primary'], 2)
        
        cv2.putText(frame, f"LEFT: {self.rep_count_left} | RIGHT: {self.rep_count_right}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        cv2.putText(frame, f"STAGE: {self.stage}", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        
        # Metrics Visualization
        # Knee Angle
        cv2.putText(frame, f"Knee Angle: {int(self.current_metrics['knee_angle'])}°", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        # Depth Progress Bar
        depth_progress = min(100, self.current_metrics['depth_percent'])
        cv2.putText(frame, f"Depth: {int(depth_progress)}%", (40, 200), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.rectangle(frame, (40, 210), (340, 225), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 210), (40 + int(3.0 * depth_progress), 225), status_color, -1)
        
        # Knee Alignment Indicator
        alignment_color = self.colors['success'] if self.current_metrics['knee_alignment'] == 'Good' else self.colors['error']
        cv2.putText(frame, f"Knee: {self.current_metrics['knee_alignment']}", (40, 250), cv2.FONT_HERSHEY_DUPLEX, 0.6, alignment_color, 1)

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
        """Process pose landmarks for lateral lunge analysis"""
        def get_coords(idx):
            lm = landmarks[idx]
            return (int(lm.x * w), int(lm.y * h))
        
        def get_coords_norm(idx):
            lm = landmarks[idx]
            return (lm.x, lm.y)

        # Landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_HEEL = 29
        RIGHT_HEEL = 30

        # Get coordinates
        l_hip = get_coords(LEFT_HIP)
        r_hip = get_coords(RIGHT_HIP)
        l_knee = get_coords(LEFT_KNEE)
        r_knee = get_coords(RIGHT_KNEE)
        l_ankle = get_coords(LEFT_ANKLE)
        r_ankle = get_coords(RIGHT_ANKLE)
        l_sh = get_coords(LEFT_SHOULDER)
        r_sh = get_coords(RIGHT_SHOULDER)

        # Calculate hip width (standing position reference)
        hip_width = abs(l_hip[0] - r_hip[0])
        
        # Determine which side is lunging based on knee bend
        left_knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
        
        # Check if in lunge position (one knee significantly bent)
        is_left_lunge = left_knee_angle < 140 and right_knee_angle > 160
        is_right_lunge = right_knee_angle < 140 and left_knee_angle > 160
        
        # Calculate depth (how low the hip is)
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2
        mid_knee_y = (l_knee[1] + r_knee[1]) / 2
        
        if is_left_lunge:
            working_knee_angle = left_knee_angle
            working_knee = l_knee
            working_hip = l_hip
            working_ankle = l_ankle
            
            # Check knee alignment (should track over toes)
            knee_over_ankle_x = abs(working_knee[0] - working_ankle[0])
            knee_alignment = "Good" if knee_over_ankle_x < 50 else "Valgus Warning"
            
            # Calculate depth percentage (90 degrees = 100%)
            depth_percent = max(0, (180 - working_knee_angle) / 90 * 100)
            
            self.current_metrics['knee_angle'] = working_knee_angle
            self.current_metrics['knee_alignment'] = knee_alignment
            self.current_metrics['depth_percent'] = depth_percent
            
            if self.stage != "Lunging Left":
                self.stage = "Lunging Left"
                self.current_side = "left"
            
            # Provide feedback
            if knee_alignment != "Good":
                self.feedback = "Alert: Keep knee aligned over toes. Don't let it collapse inward!"
            elif working_knee_angle < 70:
                self.feedback = "Warning: Don't go too deep. Risk of injury."
            elif working_knee_angle > 110:
                self.feedback = "Keep going down. Aim for 90° knee bend."
            elif 80 <= working_knee_angle <= 100:
                self.feedback = "Perfect depth! Push through heel to return."
            else:
                self.feedback = "Great form! Maintain neutral spine."
                
        elif is_right_lunge:
            working_knee_angle = right_knee_angle
            working_knee = r_knee
            working_hip = r_hip
            working_ankle = r_ankle
            
            # Check knee alignment
            knee_over_ankle_x = abs(working_knee[0] - working_ankle[0])
            knee_alignment = "Good" if knee_over_ankle_x < 50 else "Valgus Warning"
            
            # Calculate depth percentage
            depth_percent = max(0, (180 - working_knee_angle) / 90 * 100)
            
            self.current_metrics['knee_angle'] = working_knee_angle
            self.current_metrics['knee_alignment'] = knee_alignment
            self.current_metrics['depth_percent'] = depth_percent
            
            if self.stage != "Lunging Right":
                self.stage = "Lunging Right"
                self.current_side = "right"
            
            # Provide feedback
            if knee_alignment != "Good":
                self.feedback = "Alert: Keep knee aligned over toes. Don't let it collapse inward!"
            elif working_knee_angle < 70:
                self.feedback = "Warning: Don't go too deep. Risk of injury."
            elif working_knee_angle > 110:
                self.feedback = "Keep going down. Aim for 90° knee bend."
            elif 80 <= working_knee_angle <= 100:
                self.feedback = "Perfect depth! Push through heel to return."
            else:
                self.feedback = "Great form! Maintain neutral spine."
        else:
            # Standing position
            if self.stage == "Lunging Left":
                self.rep_count_left += 1
                self.stage = "Standing"
                self.feedback = f"Left rep complete! Total: {self.rep_count_left}"
                self.current_side = None
            elif self.stage == "Lunging Right":
                self.rep_count_right += 1
                self.stage = "Standing"
                self.feedback = f"Right rep complete! Total: {self.rep_count_right}"
                self.current_side = None
            else:
                self.stage = "Standing"
                self.feedback = "Ready. Step to the side and lunge."
            
            self.current_metrics['knee_angle'] = 180
            self.current_metrics['knee_alignment'] = 'Good'
            self.current_metrics['depth_percent'] = 0

    def save_report(self, frame):
        """Saves a 'Poster Analysis' summary of the session."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"lateral_lunge_analysis_{timestamp}.png"
        
        # Create a report canvas
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        
        # Add a footer area with details
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "LATERAL LUNGE POSTURE ANALYSIS REPORT", (w // 2 - 280, h + 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        
        cv2.putText(report, f"Left Reps: {self.rep_count_left} | Right Reps: {self.rep_count_right}", (50, h + 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        cv2.putText(report, f"Knee Alignment: {self.current_metrics['knee_alignment']}", 
                    (50, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        cv2.imwrite(filename, report)
        print(f"Analysis saved as {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = LateralLungeAnalyzer()

    # Set resolution for better UI representation
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("--- Lateral Lunge Analysis Started ---")
    print("Press 's' to save a Report Poster")
    print("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Premium Lateral Lunge Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            analyzer.save_report(processed_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
