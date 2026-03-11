import cv2
import numpy as np
import time
import math

class BirdDogAnalyzer:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        
        # Try to initialize Legacy Pose (MediaPipe Solutions)
        try:
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
            print("Using MediaPipe Legacy Solutions")
        except (ImportError, AttributeError):
            # Fallback to MediaPipe Tasks API
            print("MediaPipe Solutions not found or incompatible. Using Tasks API...")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request
            import os
            
            self.mp_pose = None # Not used in Tasks mode
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
        self.stage = "Neutral" # Neutral, Extending, Hold, Returning
        self.feedback = "Get on all fours. Extend opposite arm and leg."
        self.rep_count = 0
        self.hold_start_time = None
        self.required_hold_time = 2.0 # seconds
        
        self.current_metrics = {
            'arm_angle': 0,
            'leg_angle': 0,
            'back_angle': 0,
            'arm_extension': 0, # % extension
            'leg_extension': 0,  # % extension
            'is_extended': False,
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
        cv2.rectangle(overlay, (20, 20), (380, 380), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Header
        cv2.putText(frame, "BIRD DOG ANALYSIS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors['text'], 2)
        cv2.line(frame, (40, 65), (360, 65), self.colors['primary'], 2)
        
        # Stats
        cv2.putText(frame, f"REPS: {self.rep_count}", (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        cv2.putText(frame, f"STAGE: {self.stage}", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['secondary'], 1)
        
        # Metrics Visualization
        # 1. Arm Extension
        arm_ext = self.current_metrics['arm_extension']
        arm_color = self.colors['success'] if arm_ext > 80 else self.colors['warning']
        cv2.putText(frame, f"Arm Extension: {int(arm_ext)}%", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.rectangle(frame, (40, 180), (340, 195), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 180), (40 + int(3.0 * arm_ext), 195), arm_color, -1)
        
        # 2. Leg Extension
        leg_ext = self.current_metrics['leg_extension']
        leg_color = self.colors['success'] if leg_ext > 80 else self.colors['warning']
        cv2.putText(frame, f"Leg Extension: {int(leg_ext)}%", (40, 230), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.rectangle(frame, (40, 240), (340, 255), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 240), (40 + int(3.0 * leg_ext), 255), leg_color, -1)

        # 3. Back Alignment
        back_angle = self.current_metrics['back_angle']
        # Back should be relatively horizontal (90 deg relative to vertical or 0/180 relative to horizontal)
        # In our calculation, let's say 180 is perfectly flat.
        back_dev = abs(180 - back_angle)
        back_color = self.colors['success'] if back_dev < 15 else self.colors['error']
        cv2.putText(frame, f"Back Stability: {int(100 - back_dev*2)}%", (40, 290), cv2.FONT_HERSHEY_DUPLEX, 0.6, back_color, 1)

        # Progress till hold complete
        if self.stage == "Hold" and self.hold_start_time:
            elapsed = time.time() - self.hold_start_time
            hold_progress = min(100, (elapsed / self.required_hold_time) * 100)
            cv2.rectangle(frame, (40, 310), (340, 325), (50, 50, 50), -1)
            cv2.rectangle(frame, (40, 310), (40 + int(3.0 * hold_progress), 325), self.colors['secondary'], -1)
            cv2.putText(frame, "HOLDING...", (140, 322), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Form Status
        status_color = self.colors['success'] if "Good" in self.current_metrics['form_status'] else self.colors['warning']
        cv2.putText(frame, f"Status: {self.current_metrics['form_status']}", (40, 360), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1)

        # Feedback Box
        cv2.rectangle(overlay, (20, h-80), (w-20, h-20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"FEEDBACK: {self.feedback}", (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

    def draw_skeleton(self, frame, landmarks, w, h):
        connections = [
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16), # Upper body
            (11, 23), (12, 24), (23, 24), # Torso
            (23, 25), (24, 26), (25, 27), (26, 28) # Lower body
        ]
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                # Check visibility if reachable (for Task API landmarks is a list of objects with x,y,z,visibility)
                vis_start = getattr(start, 'visibility', 1.0)
                vis_end = getattr(end, 'visibility', 1.0)
                if vis_start > 0.5 and vis_end > 0.5:
                    cv2.line(frame, (int(start.x * w), int(start.y * h)), (int(end.x * w), int(end.y * h)), self.colors['primary'], 2)
        
        for lm in landmarks:
            if lm.visibility > 0.5:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, self.colors['secondary'], -1)

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
            # self.draw_skeleton(frame, processed_landmarks, w, h)
            
            def get_pt(idx):
                return (processed_landmarks[idx].x * w, processed_landmarks[idx].y * h)

            # Important landmarks for Bird Dog:
            # Shoulders: 11 (L), 12 (R)
            # Hips: 23 (L), 24 (R)
            # Elbows: 13 (L), 14 (R)
            # Wrists: 15 (L), 16 (R)
            # Knees: 25 (L), 26 (R)
            # Ankles: 27 (L), 28 (R)

            # Check for Left Arm / Right Leg extension
            la_ext = self.calculate_angle(get_pt(11), get_pt(13), get_pt(15)) # Shoulder, Elbow, Wrist
            rl_ext = self.calculate_angle(get_pt(24), get_pt(26), get_pt(28)) # Hip, Knee, Ankle
            
            # Check for Right Arm / Left Leg extension
            ra_ext = self.calculate_angle(get_pt(12), get_pt(14), get_pt(16))
            ll_ext = self.calculate_angle(get_pt(23), get_pt(25), get_pt(27))

            # Arm straightness (180 is straight)
            # Leg straightness (180 is straight)
            
            # Back flatness: Measure the angle of the line connecting shoulder and hip relative to the horizontal
            shoulder_avg = ((processed_landmarks[11].x + processed_landmarks[12].x) / 2 * w, 
                          (processed_landmarks[11].y + processed_landmarks[12].y) / 2 * h)
            hip_avg = ((processed_landmarks[23].x + processed_landmarks[24].x) / 2 * w, 
                      (processed_landmarks[23].y + processed_landmarks[24].y) / 2 * h)
            
            # Vector from Hip to Shoulder
            dx = shoulder_avg[0] - hip_avg[0]
            dy = shoulder_avg[1] - hip_avg[1]
            
            # Angle relative to horizontal (180 degrees if flat, or 0 if flat depending on orientation)
            # We want to see how close it is to horizontal (dy=0)
            back_angle_rel = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))
            back_stability_score = max(0, min(100, 100 - (back_angle_rel * 2))) # 0 deg deviation = 100%
            
            self.current_metrics['back_angle'] = 180 - back_angle_rel # Map to 180 for "flat"
            
            # Determine which side is lifting
            # If left arm and right leg are higher than their neutral counterparts
            is_l_arm_raised = processed_landmarks[15].y < processed_landmarks[11].y # Wrist higher than shoulder
            is_r_leg_raised = processed_landmarks[28].y < processed_landmarks[24].y # Ankle higher than hip
            
            is_r_arm_raised = processed_landmarks[16].y < processed_landmarks[12].y
            is_l_leg_raised = processed_landmarks[27].y < processed_landmarks[23].y

            active_arm_ext = 0
            active_leg_ext = 0
            
            if (is_l_arm_raised or is_r_leg_raised):
                active_arm_ext = la_ext
                active_leg_ext = rl_ext
                # Extension quality: How close to 180 is the limb, AND is it horizontal?
                arm_reach_y = abs(processed_landmarks[15].y - processed_landmarks[11].y)
                leg_reach_y = abs(processed_landmarks[28].y - processed_landmarks[24].y)
                # Map to %
                arm_score = max(0, min(100, (1 - arm_reach_y/0.1) * 100)) # Closer to 0 Y diff is better
                leg_score = max(0, min(100, (1 - leg_reach_y/0.1) * 100))
            elif (is_r_arm_raised or is_l_leg_raised):
                active_arm_ext = ra_ext
                active_leg_ext = ll_ext
                arm_reach_y = abs(processed_landmarks[16].y - processed_landmarks[12].y)
                leg_reach_y = abs(processed_landmarks[27].y - processed_landmarks[23].y)
                arm_score = max(0, min(100, (1 - arm_reach_y/0.1) * 100))
                leg_score = max(0, min(100, (1 - leg_reach_y/0.1) * 100))
            else:
                arm_score = 0
                leg_score = 0

            self.current_metrics['arm_extension'] = arm_score
            self.current_metrics['leg_extension'] = leg_score
            self.current_metrics['back_angle'] = 180 # Placeholder for now

            # Logic Flow
            EXT_THRESHOLD = 70
            
            if self.stage == "Neutral":
                if arm_score > 50 or leg_score > 50:
                    self.stage = "Extending"
                    self.feedback = "Maintain balance and extend fully."
                    
            elif self.stage == "Extending":
                if arm_score > EXT_THRESHOLD and leg_score > EXT_THRESHOLD:
                    self.stage = "Hold"
                    self.hold_start_time = time.time()
                    self.feedback = "Hold it! Keep your core tight."
                elif arm_score < 30 and leg_score < 30:
                    self.stage = "Neutral"
                    
            elif self.stage == "Hold":
                if arm_score > EXT_THRESHOLD and leg_score > EXT_THRESHOLD:
                    elapsed = time.time() - self.hold_start_time
                    if elapsed >= self.required_hold_time:
                        self.rep_count += 1
                        self.stage = "Returning"
                        self.feedback = "Great hold! Now return slowly."
                        self.current_metrics['form_status'] = 'Good Rep'
                else:
                    self.stage = "Extending"
                    self.feedback = "Hold was lost. Get back into position."
                    
            elif self.stage == "Returning":
                if arm_score < 30 and leg_score < 30:
                    self.stage = "Neutral"
                    self.feedback = "Switch sides or repeat."
                    self.current_metrics['form_status'] = 'Waiting'

            # self.draw_premium_ui(frame)
            self.draw_skeleton(frame, processed_landmarks, w, h)

        return frame

    def save_report(self, frame):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"bird_dog_analysis_{timestamp}.png"
        h, w, _ = frame.shape
        report = np.zeros((h + 200, w, 3), dtype=np.uint8)
        report[:h, :w] = frame
        cv2.rectangle(report, (0, h), (w, h + 200), (30, 30, 30), -1)
        cv2.putText(report, "BIRD DOG EXERCISE REPORT", (w // 2 - 250, h + 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.colors['primary'], 2)
        cv2.putText(report, f"Total Reps: {self.rep_count}", (50, h + 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, self.colors['text'], 1)
        cv2.putText(report, f"Date: {time.ctime()}", (w - 400, h + 140), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.colors['text'], 1)
        cv2.imwrite(filename, report)
        print(f"Report saved: {filename}")

def main():
    cap = cv2.VideoCapture(0)
    analyzer = BirdDogAnalyzer()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow('Premium Bird Dog Analysis', cv2.WINDOW_NORMAL)
    
    print("--- Bird Dog Analysis Started ---")
    print("Press 'q' to quit, 's' to save report")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        processed_frame = analyzer.analyze(frame.copy())
        
        cv2.imshow('Premium Bird Dog Analysis', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('s') or key == ord('S'):
            analyzer.save_report(processed_frame)
            
        if cv2.getWindowProperty('Premium Bird Dog Analysis', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    analyzer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
