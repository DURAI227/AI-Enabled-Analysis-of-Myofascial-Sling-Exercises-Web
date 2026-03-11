import cv2
import numpy as np
import time
import math

class TrunkRotationAnalyzer:
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
                enable_segmentation=False,
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
            'warning': (0, 165, 255),    # Gold
            'hip_line': (200, 50, 200)   # Purple for hip ref
        }
        
        # State tracking
        self.stage = "Neutral"  # Neutral, Twisting Left, Twisting Right
        self.feedback = "Stand straight. Keep hips stable while rotating shoulders."
        self.rep_count = 0
        self.max_rotation = 0
        self.direction = None # 'left' or 'right'
        self.current_metrics = {
            'shoulder_rotation': 0,
            'hip_rotation': 0,
            'torso_twist': 0,
            'form_status': 'Good'
        }

    def close(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()

    def calculate_transverse_rotation(self, p1, p2):
        """
        Calculate rotation in the transverse plane (top-down view).
        p1: Left point (e.g. Left Shoulder)
        p2: Right point (e.g. Right Shoulder)
        Returns angle in degrees. 0 is straight facing camera.
        Positive is Left Shoulder Forward (Turn Right).
        Negative is Right Shoulder Forward (Turn Left).
        Using Z coordinate (depth) relative to X.
        """
        # MediaPipe World Coordinates:
        # x: Right (+), Left (-)
        # y: Down (+), Up (-)
        # z: Camera -> Subject (+) -- actually relative to mid-hip origin usually
        
        # Vector from Right (p2) to Left (p1)
        dx = p1.x - p2.x
        dz = p1.z - p2.z
        
        # atan2(y, x) -> atan2(dz, dx)
        # If facing camera (dx large, dz small) -> angle ~ 0
        # If turned 90 deg right (Left shoulder back +Z, Right forward -Z) -> dz > 0, dx ~ 0 -> +90
        return np.degrees(np.arctan2(dz, dx))

    def draw_premium_ui(self, frame):
        h, w, _ = frame.shape
        # Create a semi-transparent overlay
        overlay = frame.copy()
        
        # Dashboard Background
        cv2.rectangle(overlay, (20, 20), (380, 420), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Header
        cv2.putText(frame, "TRUNK ROTATION ANALYSIS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (360, 65), self.colors['primary'], 2)
        
        # Stats
        cv2.putText(frame, f"REPS: {self.rep_count}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        cv2.putText(frame, f"STAGE: {self.stage}", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        # Metrics
        s_rot = self.current_metrics['shoulder_rotation']
        h_rot = self.current_metrics['hip_rotation']
        twist = self.current_metrics['torso_twist']
        
        # Shoulder Rotation
        cv2.putText(frame, f"Shoulder Rot: {int(s_rot)}°", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['primary'], 1)
        
        # Hip Rotation (Should be low)
        hip_color = self.colors['success'] if abs(h_rot) < 15 else self.colors['error']
        cv2.putText(frame, f"Hip Rot: {int(h_rot)}°", (40, 200), cv2.FONT_HERSHEY_DUPLEX, 0.6, hip_color, 1)
        
        # Net Twist
        cv2.putText(frame, f"Net Twist: {int(twist)}°", (40, 230), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['secondary'], 1)
        
        # Visualization (Top-Down View Simulation)
        center_x, center_y = 200, 330
        radius = 50
        cv2.circle(frame, (center_x, center_y), radius, (60, 60, 60), 2)
        cv2.putText(frame, "TOP VIEW", (center_x - 40, center_y - 60), cv2.FONT_HERSHEY_PLAIN, 1, self.colors['text'], 1)
        
        # Draw Shoulder Line (Orange)
        s_rad = np.radians(s_rot)
        sx1 = int(center_x - radius * np.cos(s_rad))
        sy1 = int(center_y - radius * np.sin(s_rad))
        sx2 = int(center_x + radius * np.cos(s_rad))
        sy2 = int(center_y + radius * np.sin(s_rad))
        cv2.line(frame, (sx1, sy1), (sx2, sy2), self.colors['primary'], 3)
        
        # Draw Hip Line (Green/Purple)
        h_rad = np.radians(h_rot)
        hx1 = int(center_x - (radius-10) * np.cos(h_rad))
        hy1 = int(center_y - (radius-10) * np.sin(h_rad))
        hx2 = int(center_x + (radius-10) * np.cos(h_rad))
        hy2 = int(center_y + (radius-10) * np.sin(h_rad))
        cv2.line(frame, (hx1, hy1), (hx2, hy2), self.colors['hip_line'], 2)

        # Form Status
        status_color = self.colors['success'] if self.current_metrics['form_status'] == 'Good' else self.colors['warning']
        if self.current_metrics['form_status'] == 'Poor': status_color = self.colors['error']
        
        cv2.putText(frame, f"Form: {self.current_metrics['form_status']}", (40, 400), cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 1)

        # Feedback Box
        cv2.rectangle(overlay, (20, h-80), (w-20, h-20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"FEEDBACK: {self.feedback}", (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

    def draw_skeleton(self, frame, landmarks, w, h):
        """Draw skeleton overlay on frame"""
        connections = [
            (11, 12), (11, 13), (12, 14), (11, 23), (12, 24), (23, 24)
        ]
        
        # Landmarks to draw
        indices = [11, 12, 23, 24] # Shoulders and Hips
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                # Filter by visibility if available (legacy only has explicit visibility)
                # Assuming visible for simplicity or checks inside library
                
                # Draw lines
                cv2.line(frame, 
                         (int(start.x * w), int(start.y * h)), 
                         (int(end.x * w), int(end.y * h)), 
                         self.colors['primary'], 2)
        
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, self.colors['secondary'], -1)

    def analyze(self, frame):
        h, w, _ = frame.shape
        
        # Frame processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.use_legacy:
            results = self.pose.process(frame_rgb)
            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
            world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            self.frame_timestamp += 1
            results = self.pose.detect_for_video(mp_image, self.frame_timestamp)
            landmarks = results.pose_landmarks[0] if results.pose_landmarks else None
            world_landmarks = results.pose_world_landmarks[0] if results.pose_world_landmarks else None

        if landmarks and world_landmarks:
            # Draw basic skeleton (2D)
            self.draw_skeleton(frame, landmarks, w, h)
            
            # Process rotation using 3D world landmarks
            # Indices: Left Shoulder 11, Right Shoulder 12, Left Hip 23, Right Hip 24
            
            # 1. Calculate Shoulder Rotation
            shoulder_rot = self.calculate_transverse_rotation(world_landmarks[11], world_landmarks[12])
            
            # 2. Calculate Hip Rotation
            hip_rot = self.calculate_transverse_rotation(world_landmarks[23], world_landmarks[24])
            
            # 3. Calculate Net Twist (Shoulder relative to Hip)
            # Both should have comparable signs. Try difference.
            # If S=30, H=5 -> Twist=25
            # If S=-30, H=-5 -> Twist=-25
            twist = shoulder_rot - hip_rot
            
            self.current_metrics['shoulder_rotation'] = shoulder_rot
            self.current_metrics['hip_rotation'] = hip_rot
            self.current_metrics['torso_twist'] = abs(twist)
            
            # Logic Flow
            abs_twist = abs(twist)
            abs_hip = abs(hip_rot)
            
            # Check Hip Stability (First priority)
            if abs_hip > 20:
                self.feedback = "Keep hips stable! Rotate from torso only."
                self.current_metrics['form_status'] = 'Poor'
            elif abs_twist > 40:
                self.feedback = "Great range of motion!"
                self.current_metrics['form_status'] = 'Excellent'
            elif abs_twist > 15:
                # Active Rotation
                self.current_metrics['form_status'] = 'Good'
                if twist > 0:
                    self.feedback = "Twisting Right..."
                    if self.stage != "Twisting Right":
                         self.stage = "Twisting Right"
                else:
                    self.feedback = "Twisting Left..."
                    if self.stage != "Twisting Left":
                         self.stage = "Twisting Left"
            else:
                # Neutral
                if self.stage != "Neutral":
                    # Completed a rep?
                    if self.stage in ["Twisting Left", "Twisting Right"]:
                        if self.max_rotation > 30: # Only count if sufficient ROM
                            self.rep_count += 1
                            self.feedback = "Rep Complete! Return to center."
                        self.max_rotation = 0
                    self.stage = "Neutral"
                self.feedback = "Stand tall. Rotate shoulders left or right."
                
            # Track max rotation during movement
            if self.stage != "Neutral":
                self.max_rotation = max(self.max_rotation, abs_twist)

        # self.draw_premium_ui(frame)
        return frame

    def save_report(self, frame):
        """Saves a summary report."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"trunk_rotation_analysis_{timestamp}.png"
        
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "TRUNK ROTATION ANALYSIS REPORT", (w // 2 - 250, h + 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        
        cv2.putText(report, f"Total Reps: {self.rep_count}", (50, h + 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        cv2.imwrite(filename, report)
        print(f"Analysis saved as {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = TrunkRotationAnalyzer()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow('Premium Trunk Rotation Analysis', cv2.WINDOW_NORMAL)

    print("--- Trunk Rotation Analysis Started ---")
    print("Press 's' to save a Report Poster")
    print("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Premium Trunk Rotation Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Quit command received.")
            break
        elif key == ord('s') or key == ord('S'):
            analyzer.save_report(processed_frame)
            
        if cv2.getWindowProperty('Premium Trunk Rotation Analysis', cv2.WND_PROP_VISIBLE) < 1:
             print("Window closed.")
             break

    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
