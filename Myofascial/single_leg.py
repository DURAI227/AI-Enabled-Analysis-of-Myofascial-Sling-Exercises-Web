import cv2
import numpy as np
import math
import time

class SingleLegGluteBridgeAnalyzer:
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
        self.stage = "Starting"  # Starting, Bridge Up Left, Bridge Up Right, Lowered
        self.feedback = "Lie on your back, knees bent, feet flat on floor."
        self.rep_count_left = 0
        self.rep_count_right = 0
        self.current_side = None
        self.hold_timer = None
        self.hold_duration = 0
        self.required_hold = 1.5  # seconds
        self.current_metrics = {
            'hip_extension_angle': 0,
            'hip_level': 'Level',
            'spine_alignment': 'Neutral',
            'extension_percent': 0
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

    def draw_premium_ui(self, frame):
        h, w, _ = frame.shape
        # Create a semi-transparent overlay for the dashboard
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (420, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status Bar
        status_color = self.colors['success'] if "Perfect" in self.feedback or "Great" in self.feedback else self.colors['primary']
        if "Alert" in self.feedback or "Warning" in self.feedback:
            status_color = self.colors['error']
        elif "Keep" in self.feedback or "Hold" in self.feedback:
            status_color = self.colors['warning']
            
        cv2.putText(frame, "SINGLE LEG GLUTE BRIDGE", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (400, 65), self.colors['primary'], 2)
        
        cv2.putText(frame, f"LEFT: {self.rep_count_left} | RIGHT: {self.rep_count_right}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        cv2.putText(frame, f"STAGE: {self.stage}", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        
        # Metrics Visualization
        # Hip Extension Angle
        cv2.putText(frame, f"Hip Extension: {int(self.current_metrics['hip_extension_angle'])}°", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        # Extension Progress Bar
        extension_progress = min(100, self.current_metrics['extension_percent'])
        cv2.putText(frame, f"Extension: {int(extension_progress)}%", (40, 200), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.rectangle(frame, (40, 210), (380, 225), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 210), (40 + int(3.4 * extension_progress), 225), status_color, -1)
        
        # Hip Level Indicator
        hip_color = self.colors['success'] if self.current_metrics['hip_level'] == 'Level' else self.colors['error']
        cv2.putText(frame, f"Hips: {self.current_metrics['hip_level']}", (40, 250), cv2.FONT_HERSHEY_DUPLEX, 0.6, hip_color, 1)
        
        # Spine Alignment
        spine_color = self.colors['success'] if self.current_metrics['spine_alignment'] == 'Neutral' else self.colors['error']
        cv2.putText(frame, f"Spine: {self.current_metrics['spine_alignment']}", (40, 280), cv2.FONT_HERSHEY_DUPLEX, 0.6, spine_color, 1)

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
        """Process pose landmarks for single leg glute bridge analysis"""
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
        l_sh = get_coords(LEFT_SHOULDER)
        r_sh = get_coords(RIGHT_SHOULDER)
        l_hip = get_coords(LEFT_HIP)
        r_hip = get_coords(RIGHT_HIP)
        l_knee = get_coords(LEFT_KNEE)
        r_knee = get_coords(RIGHT_KNEE)
        l_ankle = get_coords(LEFT_ANKLE)
        r_ankle = get_coords(RIGHT_ANKLE)

        # Calculate mid points
        mid_sh = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
        mid_hip = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)

        # Check hip level (should be even) - more forgiving threshold
        hip_level_diff = abs(l_hip[1] - r_hip[1])
        hip_level = "Level" if hip_level_diff < 50 else "Tilted"  # Increased from 30 to 50
        self.current_metrics['hip_level'] = hip_level

        # Calculate hip extension angles for both sides
        left_hip_angle = self.calculate_angle(l_sh, l_hip, l_knee)
        right_hip_angle = self.calculate_angle(r_sh, r_hip, r_knee)

        # Determine which leg is raised (ankle higher than knee) - more sensitive threshold
        left_leg_raised = l_ankle[1] < l_knee[1] - 20  # Reduced from 50 to 20
        right_leg_raised = r_ankle[1] < r_knee[1] - 20  # Reduced from 50 to 20

        # Check if in bridge position (hips elevated) - more sensitive threshold
        # Hip should be higher than shoulders
        hips_elevated = mid_hip[1] < mid_sh[1] + 20  # Changed from -30 to +20 (allows lower bridge)

        # Determine current state
        if hips_elevated and left_leg_raised:
            # Left leg raised, right leg working
            working_hip_angle = right_hip_angle
            
            # Calculate extension percentage (180° = full extension)
            extension_percent = (working_hip_angle / 180) * 100
            self.current_metrics['hip_extension_angle'] = working_hip_angle
            self.current_metrics['extension_percent'] = extension_percent
            
            # Check spine alignment (should not hyperextend)
            if working_hip_angle > 185:
                self.current_metrics['spine_alignment'] = 'Hyperextended'
            elif working_hip_angle < 150:
                self.current_metrics['spine_alignment'] = 'Flexed'
            else:
                self.current_metrics['spine_alignment'] = 'Neutral'
            
            if self.stage != "Bridge Up Right":
                self.stage = "Bridge Up Right"
                self.current_side = "right"
                self.hold_timer = time.time()
            
            # Calculate hold duration
            self.hold_duration = time.time() - self.hold_timer
            
            # Provide feedback
            if hip_level != "Level":
                self.feedback = "Alert: Keep hips level! Don't let them tilt to one side."
            elif self.current_metrics['spine_alignment'] == 'Hyperextended':
                self.feedback = "Warning: Don't arch your back! Engage your core."
            elif working_hip_angle < 160:
                self.feedback = "Lift hips higher. Form a straight line from shoulder to knee."
            elif self.hold_duration < self.required_hold:
                self.feedback = f"Hold position... {self.required_hold - self.hold_duration:.1f}s"
            else:
                self.feedback = "Perfect form! Squeeze glutes and hold."
                
        elif hips_elevated and right_leg_raised:
            # Right leg raised, left leg working
            working_hip_angle = left_hip_angle
            
            extension_percent = (working_hip_angle / 180) * 100
            self.current_metrics['hip_extension_angle'] = working_hip_angle
            self.current_metrics['extension_percent'] = extension_percent
            
            # Check spine alignment
            if working_hip_angle > 185:
                self.current_metrics['spine_alignment'] = 'Hyperextended'
            elif working_hip_angle < 150:
                self.current_metrics['spine_alignment'] = 'Flexed'
            else:
                self.current_metrics['spine_alignment'] = 'Neutral'
            
            if self.stage != "Bridge Up Left":
                self.stage = "Bridge Up Left"
                self.current_side = "left"
                self.hold_timer = time.time()
            
            self.hold_duration = time.time() - self.hold_timer
            
            # Provide feedback
            if hip_level != "Level":
                self.feedback = "Alert: Keep hips level! Don't let them tilt to one side."
            elif self.current_metrics['spine_alignment'] == 'Hyperextended':
                self.feedback = "Warning: Don't arch your back! Engage your core."
            elif working_hip_angle < 140:
                self.feedback = "Lift hips higher. Form a straight line from shoulder to knee."
            elif self.hold_duration < self.required_hold:
                self.feedback = f"Hold position... {self.required_hold - self.hold_duration:.1f}s"
            else:
                self.feedback = "Perfect form! Squeeze glutes and hold."
        else:
            # Lowered position
            if self.stage == "Bridge Up Left":
                if self.hold_duration >= self.required_hold:
                    self.rep_count_left += 1
                    self.feedback = f"Left rep complete! Total: {self.rep_count_left}"
                else:
                    self.feedback = "Hold longer at the top for full rep."
                self.stage = "Lowered"
                self.hold_timer = None
            elif self.stage == "Bridge Up Right":
                if self.hold_duration >= self.required_hold:
                    self.rep_count_right += 1
                    self.feedback = f"Right rep complete! Total: {self.rep_count_right}"
                else:
                    self.feedback = "Hold longer at the top for full rep."
                self.stage = "Lowered"
                self.hold_timer = None
            else:
                self.stage = "Starting"
                self.feedback = "Raise one leg and drive hips up. Squeeze glutes!"
            
            self.current_metrics['hip_extension_angle'] = 90
            self.current_metrics['extension_percent'] = 0
            self.hold_duration = 0

    def save_report(self, frame):
        """Saves a 'Poster Analysis' summary of the session."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"single_leg_glute_bridge_{timestamp}.png"
        
        # Create a report canvas
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        
        # Add a footer area with details
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "SINGLE LEG GLUTE BRIDGE ANALYSIS REPORT", (w // 2 - 320, h + 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        
        cv2.putText(report, f"Left Reps: {self.rep_count_left} | Right Reps: {self.rep_count_right}", (50, h + 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        cv2.putText(report, f"Hip Alignment: {self.current_metrics['hip_level']} | Spine: {self.current_metrics['spine_alignment']}", 
                    (50, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        cv2.imwrite(filename, report)
        print(f"Analysis saved as {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = SingleLegGluteBridgeAnalyzer()

    # Set resolution for better UI representation
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("--- Single Leg Glute Bridge Analysis Started ---")
    print("Press 's' to save a Report Poster")
    print("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Single Leg Glute Bridge Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            analyzer.save_report(processed_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
