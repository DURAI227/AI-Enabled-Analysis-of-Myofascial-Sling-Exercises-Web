import cv2
import numpy as np
import time
import math

class CableChopAnalyzer:
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
        self.stage = "Start"
        self.feedback = "Stand sideways. Grip handle with both hands high."
        self.rep_count = 0
        self.chop_path = [] # Store hand positions to visualize arc
        self.direction = "Down" # Assuming High-to-Low chop
        
        self.current_metrics = {
            'torso_rotation': 0,
            'arm_extension': 0,
            'squat_depth': 0,
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

    def calculate_transverse_rotation(self, p1, p2):
        """Estimate rotation based on Z depth difference."""
        dx = p1.x - p2.x
        dz = p1.z - p2.z
        return np.degrees(np.arctan2(dz, dx))

    def draw_premium_ui(self, frame):
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Dashboard Background
        cv2.rectangle(overlay, (20, 20), (380, 380), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Header
        cv2.putText(frame, "CABLE CHOP ANALYSIS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (360, 65), self.colors['primary'], 2)
        
        # Stats
        cv2.putText(frame, f"REPS: {self.rep_count}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        cv2.putText(frame, f"STAGE: {self.stage}", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['secondary'], 1)
        
        # Metrics Visualization
        # 1. Torso Rotation (The Power Source)
        rot = self.current_metrics['torso_rotation']
        # Display as a gauge type bar (-90 to 90)
        # Center is 0.
        bar_w = 300
        center_x = 40 + bar_w // 2
        
        # Normalize -60 to +60 degrees to pixels
        offset = int(np.clip(rot, -60, 60) / 60 * (bar_w // 2))
        
        cv2.putText(frame, f"Torso Twist: {int(rot)}°", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.line(frame, (center_x, 180), (center_x, 195), self.colors['text'], 2) # Center marker
        cv2.rectangle(frame, (center_x, 185), (center_x + offset, 195), self.colors['primary'], -1)
        cv2.rectangle(frame, (40, 185), (40 + bar_w, 195), (100, 100, 100), 1) # Border
        
        # 2. Arm Extension
        ext = self.current_metrics['arm_extension']
        ext_color = self.colors['success'] if ext > 140 else self.colors['warning']
        if ext < 100: ext_color = self.colors['error']
        cv2.putText(frame, f"Arm Extension: {int(ext)}°", (40, 230), cv2.FONT_HERSHEY_DUPLEX, 0.6, ext_color, 1)

        # Form Status
        status_color = self.colors['success'] if "Good" in self.current_metrics['form_status'] else self.colors['warning']
        if "Bad" in self.current_metrics['form_status']: status_color = self.colors['error']
        
        cv2.putText(frame, f"Status: {self.current_metrics['form_status']}", (40, 320), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1)

        # Feedback Box
        cv2.rectangle(overlay, (20, h-80), (w-20, h-20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"FEEDBACK: {self.feedback}", (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

    def draw_skeleton(self, frame, landmarks, w, h):
        # Draw chop path
        if len(self.chop_path) > 1:
            for i in range(1, len(self.chop_path)):
                pt1 = self.chop_path[i-1]
                pt2 = self.chop_path[i]
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
        
        # Standard skeleton
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
        
        # Highlight Hands
        lx, ly = int(landmarks[15].x * w), int(landmarks[15].y * h)
        rx, ry = int(landmarks[16].x * w), int(landmarks[16].y * h)
        cv2.circle(frame, (lx, ly), 8, self.colors['secondary'], -1)
        cv2.circle(frame, (rx, ry), 8, self.colors['secondary'], -1)

    def analyze(self, frame):
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        processed_landmarks = None
        world_landmarks = None
        
        if self.use_legacy:
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                processed_landmarks = results.pose_landmarks.landmark
                # Legacy world landmarks often unstable or unavailable depending on version
                if hasattr(results, 'pose_world_landmarks') and results.pose_world_landmarks:
                    world_landmarks = results.pose_world_landmarks.landmark
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            self.frame_timestamp += 1
            results = self.pose.detect_for_video(mp_image, self.frame_timestamp)
            if results.pose_landmarks:
                processed_landmarks = results.pose_landmarks[0]
                world_landmarks = results.pose_world_landmarks[0]

        if processed_landmarks:
            self.draw_skeleton(frame, processed_landmarks, w, h)
            
            def get_pt(idx):
                return (int(processed_landmarks[idx].x * w), int(processed_landmarks[idx].y * h))
            
            # --- Metrics ---
            # 1. Arm Straightness (Average of both elbows)
            l_angle = self.calculate_angle(
                (processed_landmarks[11].x, processed_landmarks[11].y),
                (processed_landmarks[13].x, processed_landmarks[13].y),
                (processed_landmarks[15].x, processed_landmarks[15].y)
            )
            r_angle = self.calculate_angle(
                (processed_landmarks[12].x, processed_landmarks[12].y),
                (processed_landmarks[14].x, processed_landmarks[14].y),
                (processed_landmarks[16].x, processed_landmarks[16].y)
            )
            avg_arm_ext = (l_angle + r_angle) / 2
            
            # 2. Torso Rotation
            rot_deg = 0
            if world_landmarks:
                rot_deg = self.calculate_transverse_rotation(world_landmarks[11], world_landmarks[12])
            
            # 3. Hand Height (relative to Hip height for Phase detection)
            # Find average hand Y
            hands_y = (processed_landmarks[15].y + processed_landmarks[16].y) / 2
            # Hip Y
            hips_y = (processed_landmarks[23].y + processed_landmarks[24].y) / 2
            # Normalized Height (0 is top, 1 is bottom) -> > hip means below hip
            
            # Store path point roughly
            center_hand = ((get_pt(15)[0] + get_pt(16)[0]) // 2, (get_pt(15)[1] + get_pt(16)[1]) // 2)
            self.chop_path.append(center_hand)
            if len(self.chop_path) > 30: self.chop_path.pop(0)
            
            # State Machine
            UP_THRESH = hips_y - 0.3 # Above hips considerably (approx shoulder/head level)
            DOWN_THRESH = hips_y # At or below hips
            
            # Check Form
            is_good_form = True
            if avg_arm_ext < 120:
                self.feedback = "Keep arms straighter!"
                self.current_metrics['form_status'] = 'Bend Arms'
                is_good_form = False
            else:
                self.current_metrics['form_status'] = 'Good'

            if self.stage == "Start":
                if hands_y < UP_THRESH: # Hands High
                    self.stage = "High"
                    self.feedback = "Chop down across body!"
                    self.chop_path = [] # Reset path on new start usually
            
            elif self.stage == "High":
                if hands_y > DOWN_THRESH: # Hands Low
                    if is_good_form:
                        self.stage = "Low"
                        self.feedback = "Good Chop! Return slowly."
                        self.rep_count += 1
                        self.current_metrics['form_status'] = 'Good Rep'
                    else:
                        self.feedback = "Rep incomplete form."
            
            elif self.stage == "Low":
                if hands_y < UP_THRESH: # Returned to top
                    self.stage = "High"
                    self.feedback = "Ready for next rep."
                    self.chop_path = [] # Reset path
            
            self.current_metrics['torso_rotation'] = rot_deg
            self.current_metrics['arm_extension'] = avg_arm_ext

        # self.draw_premium_ui(frame)
        return frame

    def save_report(self, frame):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"cable_chop_analysis_{timestamp}.png"
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "CABLE CHOP REPORT", (w // 2 - 250, h + 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        cv2.putText(report, f"Reps: {self.rep_count}", (50, h + 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.imwrite(filename, report)
        print(f"Report saved: {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = CableChopAnalyzer()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Premium Cable Chop Analysis', cv2.WINDOW_NORMAL)
    
    print("--- Cable Chop Analysis Started ---")
    print("High-to-Low or Low-to-High supported (Detects vertical travel)")
    print("Press 'q' to quit, 's' to save report")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Premium Cable Chop Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Quit command received.")
            break
        elif key == ord('s') or key == ord('S'):
            analyzer.save_report(processed_frame)
            
        if cv2.getWindowProperty('Premium Cable Chop Analysis', cv2.WND_PROP_VISIBLE) < 1:
             print("Window closed.")
             break

    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
