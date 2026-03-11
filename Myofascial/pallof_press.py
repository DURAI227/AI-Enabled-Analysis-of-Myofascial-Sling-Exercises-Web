import cv2
import numpy as np
import time
import math

class PallofPressAnalyzer:
    def __init__(self):
        # MediaPipe Initialization
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
            'center_line': (100, 100, 255) # Light Red for target
        }
        
        # State tracking
        self.stage = "Start"  # Start (at chest), Extended, Returning
        self.feedback = "Stand sideways. Press hands straight out."
        self.rep_count = 0
        self.max_extension = 0
        self.current_metrics = {
            'extension_ratio': 0,
            'deviation_x': 0,
            'rotation': 0,
            'form_status': 'Good'
        }

    def close(self):
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def calculate_transverse_rotation(self, p1, p2):
        """Estimate rotation based on Z depth difference."""
        dx = p1.x - p2.x
        dz = p1.z - p2.z
        return np.degrees(np.arctan2(dz, dx))

    def draw_premium_ui(self, frame):
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Dashboard Background
        cv2.rectangle(overlay, (20, 20), (380, 400), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Header
        cv2.putText(frame, "PALLOF PRESS ANALYSIS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (360, 65), self.colors['primary'], 2)
        
        # Stats
        cv2.putText(frame, f"REPS: {self.rep_count}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        cv2.putText(frame, f"STAGE: {self.stage}", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        
        # Metrics Visualization
        # 1. Extension Progress
        ext_ratio = self.current_metrics['extension_ratio']
        # Map roughly 0.2 (chest) to 0.8 (full extension) -> 0 to 100%
        ext_percent = max(0, min(100, (ext_ratio - 0.2) / 0.6 * 100))
        
        cv2.putText(frame, f"Extension: {int(ext_percent)}%", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.rectangle(frame, (40, 180), (340, 195), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 180), (40 + int(3.0 * ext_percent), 195), self.colors['primary'], -1)
        
        # 2. Stability / Deviation
        dev_x = self.current_metrics['deviation_x']
        # Display deviation as a centered bar
        # 0 is center (good). Limits +/- 0.2
        cv2.putText(frame, "Center Deviation", (40, 225), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        bar_center = 190
        # Scale: 0.1 deviation = 50 pixels
        dev_px = int(dev_x * 500)
        
        # Draw center mark
        cv2.line(frame, (bar_center + 40, 235), (bar_center + 40, 255), self.colors['text'], 2)
        # Draw bar from center
        bar_color = self.colors['success'] if abs(dev_x) < 0.05 else self.colors['warning']
        if abs(dev_x) > 0.1: bar_color = self.colors['error']
        
        start_x = bar_center + 40
        end_x = start_x + dev_px
        cv2.rectangle(frame, (min(start_x, end_x), 240), (max(start_x, end_x), 250), bar_color, -1)
        
        # 3. Rotation (Text only)
        rot = self.current_metrics['rotation']
        rot_color = self.colors['success'] if abs(rot) < 15 else self.colors['error']
        cv2.putText(frame, f"Rotation: {int(rot)}°", (40, 290), cv2.FONT_HERSHEY_DUPLEX, 0.6, rot_color, 1)

        # Form Status
        status_color = self.colors['success'] if self.current_metrics['form_status'] == 'Stable' else self.colors['warning']
        if self.current_metrics['form_status'] == 'Unstable': status_color = self.colors['error']
        
        cv2.putText(frame, f"Status: {self.current_metrics['form_status']}", (40, 380), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1)

        # Feedback Box
        cv2.rectangle(overlay, (20, h-80), (w-20, h-20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"FEEDBACK: {self.feedback}", (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

    def draw_skeleton(self, frame, landmarks, w, h):
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24)
        ]
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                cv2.line(frame, (int(start.x * w), int(start.y * h)), (int(end.x * w), int(end.y * h)), self.colors['primary'], 2)
        
        # Draw Hands specially
        lx, ly = int(landmarks[15].x * w), int(landmarks[15].y * h)
        rx, ry = int(landmarks[16].x * w), int(landmarks[16].y * h)
        cv2.circle(frame, (lx, ly), 6, self.colors['secondary'], -1)
        cv2.circle(frame, (rx, ry), 6, self.colors['secondary'], -1)
        
        # Draw Target Line (Vertical line through chest center)
        cx = int((landmarks[11].x + landmarks[12].x) / 2 * w)
        # cy = int((landmarks[11].y + landmarks[12].y) / 2 * h)
        cv2.line(frame, (cx, 0), (cx, h), self.colors['center_line'], 1)

    def analyze(self, frame):
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        processed_landmarks = None
        world_landmarks = None

        if self.use_legacy:
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                processed_landmarks = results.pose_landmarks.landmark
                # Legacy doesn't always have accurate world landmarks depending on model, keeping simple
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            self.frame_timestamp += 1
            results = self.pose.detect_for_video(mp_image, self.frame_timestamp)
            if results.pose_landmarks:
                processed_landmarks = results.pose_landmarks[0]
                world_landmarks = results.pose_world_landmarks[0]

        if processed_landmarks:
            # Draw Skeleton
            self.draw_skeleton(frame, processed_landmarks, w, h)
            
            # --- Metrics Calculation ---
            # Indices: 11(L_Sh), 12(R_Sh), 15(L_Wrist), 16(R_Wrist)
            l_sh = processed_landmarks[11]
            r_sh = processed_landmarks[12]
            l_wrist = processed_landmarks[15]
            r_wrist = processed_landmarks[16]
            
            # 1. Shoulder Width (Reference for normalization)
            # Use 2D distance for normalization scale
            shoulder_width = math.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            if shoulder_width == 0: shoulder_width = 0.1 # Avoid div by zero
            
            # 2. Hands Position (Midpoint)
            hands_x = (l_wrist.x + r_wrist.x) / 2
            hands_y = (l_wrist.y + r_wrist.y) / 2
            
            # 3. Chest Center (Midpoint of shoulders)
            chest_x = (l_sh.x + r_sh.x) / 2
            chest_y = (l_sh.y + r_sh.y) / 2
            
            # 4. Extension (Distance from chest to hands)
            # We care mostly about Y-distance? No, distance in plane.
            # But "Extension" in Pallof Press is distance from body.
            # In 2D view (Front), extension is harder to see if perfectly perpendicular (Z-axis).
            # BUT, usually Pallof Press is filmed from Front/Side?
            # Assuming Front view: Extension is barely visible (Z).
            # Assuming Side/Angle view: Extension is X/Y distance.
            # Wait, best view for Pallof Press detection is slightly angled or side?
            # Or if front view, hands get smaller? No.
            # Let's assume standard webcam view where user pushes hands "towards camera" or "away from camera"?
            # No, Pallof Press is horizontal push.
            # If camera is front-on, hands move away from chest in Z-axis.
            # If camera is side-on, hands move in X-axis.
            # Let's assume user stands facing camera slightly angled or hands move "out".
            # Actually, easiest metric for extension in 2D without depth is:
            # Distance between Hands and Shoulders.
            
            # Calculate 2D distance between Hands Center and Chest Center
            dist_hands_chest = math.sqrt((hands_x - chest_x)**2 + (hands_y - chest_y)**2)
            extension_ratio = dist_hands_chest / shoulder_width
            
            # 5. Deviation (Horizontal drift relative to chest center)
            # X-deviation relative to shoulder width
            deviation_x = (hands_x - chest_x) / shoulder_width
            
            # 6. Rotation (World Landmarks are best, else fallback)
            rot = 0
            if world_landmarks:
                 # Use world coordinates for rotation
                 dx = world_landmarks[11].x - world_landmarks[12].x
                 dz = world_landmarks[11].z - world_landmarks[12].z
                 rot = np.degrees(np.arctan2(dz, dx))
            
            self.current_metrics['extension_ratio'] = extension_ratio
            self.current_metrics['deviation_x'] = deviation_x
            self.current_metrics['rotation'] = rot
            
            # --- Logic Flow ---
            is_stable = abs(deviation_x) < 0.15 and abs(rot) < 20
            
            if is_stable:
                self.current_metrics['form_status'] = 'Stable'
            else:
                self.current_metrics['form_status'] = 'Unstable'
                if abs(deviation_x) >= 0.15:
                    self.feedback = "Keep hands centered! Resist the pull."
                elif abs(rot) >= 20:
                    self.feedback = "Don't rotate torso! Stay square."
            
            # Rep Counting State Machine
            # Thresholds need calibration.
            # Extension Ratio: At chest ~ 0.2-0.3. Fully extended ~ 0.8-1.5 (depending on arm length/angle)
            RETRACTED_THRESH = 0.4
            EXTENDED_THRESH = 0.8
            
            if self.stage == "Start":
                if extension_ratio > EXTENDED_THRESH:
                    if is_stable:
                        self.stage = "Extended"
                        self.feedback = "Hold... Now return slowly."
                    else:
                        self.feedback = "Straighten up before extending!"
            
            elif self.stage == "Extended":
                if extension_ratio < RETRACTED_THRESH:
                    # Completed rep
                    if is_stable:
                        self.rep_count += 1
                        self.feedback = "Good Rep! Keep it controlled."
                    else:
                        self.feedback = "Rep counted, but watch stability."
                    self.stage = "Start"
                elif not is_stable:
                    self.feedback = "Stabilize! Resist rotation."
                    
        # self.draw_premium_ui(frame)
        return frame

    def save_report(self, frame):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"pallof_press_analysis_{timestamp}.png"
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "PALLOF PRESS ANALYSIS REPORT", (w // 2 - 250, h + 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        cv2.putText(report, f"Reps: {self.rep_count} | Status: {self.current_metrics['form_status']}", (50, h + 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 1)
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.imwrite(filename, report)
        print(f"Report saved: {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = PallofPressAnalyzer()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Premium Pallof Press Analysis', cv2.WINDOW_NORMAL)
    
    print("--- Pallof Press Analysis Started ---")
    print("Stand facing camera or slightly angled.")
    print("Press 'q' to quit, 's' to save report")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Premium Pallof Press Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Quit command received.")
            break
        elif key == ord('s') or key == ord('S'):
            analyzer.save_report(processed_frame)
            
        if cv2.getWindowProperty('Premium Pallof Press Analysis', cv2.WND_PROP_VISIBLE) < 1:
             print("Window closed.")
             break

    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
