import cv2
import numpy as np
import math
import time

class ClamshellAnalyzer:
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
        self.stage = "Starting"  # Starting, Closed, Open
        self.feedback = "Lie on your side, knees bent at 45°, feet together."
        self.rep_count = 0
        self.current_metrics = {
            'knee_angle': 0,
            'hip_rotation_angle': 0,
            'hip_stability': 'Stable',
            'range_of_motion': 0
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
        cv2.rectangle(overlay, (20, 20), (420, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status Bar
        status_color = self.colors['success'] if "Perfect" in self.feedback or "Great" in self.feedback else self.colors['primary']
        if "Alert" in self.feedback or "Warning" in self.feedback:
            status_color = self.colors['error']
        elif "Keep" in self.feedback or "Hold" in self.feedback:
            status_color = self.colors['warning']
            
        cv2.putText(frame, "CLAMSHELL HIP STABILITY", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (400, 65), self.colors['primary'], 2)
        
        cv2.putText(frame, f"REPS: {self.rep_count}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        cv2.putText(frame, f"STAGE: {self.stage}", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        
        # Metrics Visualization
        # Hip Rotation Angle
        cv2.putText(frame, f"Hip Rotation: {int(self.current_metrics['hip_rotation_angle'])}°", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        # Range of Motion Progress Bar
        rom_progress = min(100, self.current_metrics['range_of_motion'])
        cv2.putText(frame, f"Range: {int(rom_progress)}%", (40, 200), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.rectangle(frame, (40, 210), (380, 225), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 210), (40 + int(3.4 * rom_progress), 225), status_color, -1)
        
        # Hip Stability Indicator
        stability_color = self.colors['success'] if self.current_metrics['hip_stability'] == 'Stable' else self.colors['error']
        cv2.putText(frame, f"Hip: {self.current_metrics['hip_stability']}", (40, 250), cv2.FONT_HERSHEY_DUPLEX, 0.6, stability_color, 1)

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
        """Process pose landmarks for clamshell analysis"""
        def get_coords(idx):
            lm = landmarks[idx]
            return (int(lm.x * w), int(lm.y * h))

        # Landmark indices
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12

        # Get coordinates
        l_hip = get_coords(LEFT_HIP)
        r_hip = get_coords(RIGHT_HIP)
        l_knee = get_coords(LEFT_KNEE)
        r_knee = get_coords(RIGHT_KNEE)
        l_ankle = get_coords(LEFT_ANKLE)
        r_ankle = get_coords(RIGHT_ANKLE)
        l_sh = get_coords(LEFT_SHOULDER)
        r_sh = get_coords(RIGHT_SHOULDER)

        # Determine which side is facing up (based on which hip is higher)
        # In side-lying position, one hip should be significantly higher than the other
        is_lying_on_left = r_hip[1] < l_hip[1] - 30  # Right hip higher (lying on left)
        is_lying_on_right = l_hip[1] < r_hip[1] - 30  # Left hip higher (lying on right)

        if is_lying_on_left:
            # Lying on left side, right leg is top leg
            top_hip = r_hip
            top_knee = r_knee
            top_ankle = r_ankle
            bottom_hip = l_hip
            bottom_knee = l_knee
            
            # Calculate knee angle (should be around 45-90 degrees)
            knee_angle = self.calculate_angle(top_hip, top_knee, top_ankle)
            self.current_metrics['knee_angle'] = knee_angle
            
            # Calculate distance between knees (measures hip rotation/abduction)
            knee_separation = abs(top_knee[0] - bottom_knee[0])
            
            # Check hip stability (hips should stay stacked, not rolling)
            hip_alignment_diff = abs(r_hip[0] - l_hip[0])
            hip_stable = hip_alignment_diff < 80  # Hips relatively aligned
            self.current_metrics['hip_stability'] = 'Stable' if hip_stable else 'Rolling'
            
            # Calculate range of motion (knee separation as percentage)
            # Typical good ROM is 30-50 pixels separation
            rom_percent = min(100, (knee_separation / 50) * 100)
            self.current_metrics['range_of_motion'] = rom_percent
            self.current_metrics['hip_rotation_angle'] = knee_separation  # Simplified metric
            
            # Determine stage
            if knee_separation > 30:  # Knees apart (open position)
                if self.stage != "Open":
                    self.stage = "Open"
                
                # Provide feedback
                if not hip_stable:
                    self.feedback = "Alert: Keep hips stacked! Don't roll backward."
                elif knee_angle < 30 or knee_angle > 100:
                    self.feedback = "Adjust knee bend to 45-90 degrees."
                elif knee_separation > 80:
                    self.feedback = "Warning: Don't open too wide. Control the movement."
                else:
                    self.feedback = "Perfect! Hold briefly, then slowly close."
            else:  # Knees together (closed position)
                if self.stage == "Open":
                    self.rep_count += 1
                    self.feedback = f"Rep complete! Total: {self.rep_count}"
                    self.stage = "Closed"
                else:
                    self.stage = "Closed"
                    self.feedback = "Slowly lift top knee, keep feet together."
                    
        elif is_lying_on_right:
            # Lying on right side, left leg is top leg
            top_hip = l_hip
            top_knee = l_knee
            top_ankle = l_ankle
            bottom_hip = r_hip
            bottom_knee = r_knee
            
            # Calculate knee angle
            knee_angle = self.calculate_angle(top_hip, top_knee, top_ankle)
            self.current_metrics['knee_angle'] = knee_angle
            
            # Calculate distance between knees
            knee_separation = abs(top_knee[0] - bottom_knee[0])
            
            # Check hip stability
            hip_alignment_diff = abs(l_hip[0] - r_hip[0])
            hip_stable = hip_alignment_diff < 80
            self.current_metrics['hip_stability'] = 'Stable' if hip_stable else 'Rolling'
            
            # Calculate range of motion
            rom_percent = min(100, (knee_separation / 50) * 100)
            self.current_metrics['range_of_motion'] = rom_percent
            self.current_metrics['hip_rotation_angle'] = knee_separation
            
            # Determine stage
            if knee_separation > 30:  # Knees apart
                if self.stage != "Open":
                    self.stage = "Open"
                
                # Provide feedback
                if not hip_stable:
                    self.feedback = "Alert: Keep hips stacked! Don't roll backward."
                elif knee_angle < 30 or knee_angle > 100:
                    self.feedback = "Adjust knee bend to 45-90 degrees."
                elif knee_separation > 80:
                    self.feedback = "Warning: Don't open too wide. Control the movement."
                else:
                    self.feedback = "Perfect! Hold briefly, then slowly close."
            else:  # Knees together
                if self.stage == "Open":
                    self.rep_count += 1
                    self.feedback = f"Rep complete! Total: {self.rep_count}"
                    self.stage = "Closed"
                else:
                    self.stage = "Closed"
                    self.feedback = "Slowly lift top knee, keep feet together."
        else:
            # Not in proper side-lying position
            self.stage = "Starting"
            self.feedback = "Lie on your side with hips stacked vertically."
            self.current_metrics['knee_angle'] = 0
            self.current_metrics['hip_rotation_angle'] = 0
            self.current_metrics['range_of_motion'] = 0
            self.current_metrics['hip_stability'] = 'Stable'

    def save_report(self, frame):
        """Saves a 'Poster Analysis' summary of the session."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"clamshell_analysis_{timestamp}.png"
        
        # Create a report canvas
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        
        # Add a footer area with details
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "CLAMSHELL HIP STABILITY ANALYSIS REPORT", (w // 2 - 320, h + 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        
        cv2.putText(report, f"Total Reps: {self.rep_count}", (50, h + 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        cv2.putText(report, f"Hip Stability: {self.current_metrics['hip_stability']} | ROM: {int(self.current_metrics['range_of_motion'])}%", 
                    (50, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        cv2.imwrite(filename, report)
        print(f"Analysis saved as {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = ClamshellAnalyzer()

    # Set resolution for better UI representation
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("--- Clamshell Hip Stability Analysis Started ---")
    print("Press 's' to save a Report Poster")
    print("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Clamshell Hip Stability Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            analyzer.save_report(processed_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
