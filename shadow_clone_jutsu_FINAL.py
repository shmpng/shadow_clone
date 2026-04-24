"""
Shadow Clone Jutsu - Main Application
Gesture + Voice activated clone effect
Compatible with MediaPipe 0.10.30+ (Python 3.13)
"""

import cv2
import numpy as np
import pickle
import speech_recognition as sr
import threading
import queue
from datetime import datetime
import os
import urllib.request

print("Loading dependencies...")
try:
    import mediapipe as mp
    print(f"✅ MediaPipe loaded (v{mp.__version__})")
except:
    print("❌ Install: pip install mediapipe")
    exit(1)

class ShadowCloneJutsu:
    def __init__(self):
        print("\n🥷 SHADOW CLONE JUTSU - INITIALIZING\n")
        
        # Setup hand detector
        print("📥 Setting up hand detector...")
        self.model_path = "hand_landmarker.task"
        self._download_hand_model()
        
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
        print("✅ Hand detector ready\n")
        
        # Setup pose detector
        print("📥 Setting up pose detector...")
        pose_path = self._download_pose_model()
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_path),
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
        print("✅ Pose detector ready\n")
        
        # Load gesture model
        print("📂 Loading trained gesture model...")
        self.gesture_model = None
        self.load_gesture_model()
        
        # Voice recognition
        print("\n🎤 Setting up voice recognition...")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice_queue = queue.Queue()
        self.listening = False
        print("✅ Voice ready\n")
        
        # State
        self.gesture_detected = False
        self.voice_command_detected = False
        self.clone_active = False
        self.clone_image = None
        self.clone_mask = None
        self.clone_start_time = None
        self.clone_duration = 5
        
        # Detection params
        self.gesture_confidence_threshold = 0.6
        self.gesture_hold_frames = 3
        self.gesture_frame_count = 0
        
        print("=" * 60)
        print("✅ ALL SYSTEMS READY!")
        print("=" * 60)
    
    def _download_hand_model(self):
        if os.path.exists(self.model_path):
            return
        print("   Downloading hand model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, self.model_path)
    
    def _download_pose_model(self):
        pose_path = "pose_landmarker.task"
        if os.path.exists(pose_path):
            return pose_path
        print("   Downloading pose model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, pose_path)
        return pose_path
    
    def load_gesture_model(self):
        if os.path.exists("gesture_model.pkl"):
            with open("gesture_model.pkl", 'rb') as f:
                self.gesture_model = pickle.load(f)
            print("✅ Gesture model loaded!")
        else:
            print("❌ gesture_model.pkl not found!")
            print("Please run: python train_gesture_model.py")
            exit(1)
    
    def extract_hand_features(self, hand_landmarks):
        features = []
        for landmark in hand_landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])
        return features
    
    def detect_gesture(self, detection_result):
        if not detection_result.hand_landmarks or self.gesture_model is None:
            self.gesture_frame_count = 0
            return False
        
        all_features = []
        for hand_landmarks in detection_result.hand_landmarks:
            features = self.extract_hand_features(hand_landmarks)
            all_features.extend(features)
        
        try:
            features_array = np.array(all_features).reshape(1, -1)
            prediction = self.gesture_model.predict(features_array)[0]
            probability = self.gesture_model.predict_proba(features_array)[0]
            
            if prediction == 1 and probability[1] > self.gesture_confidence_threshold:
                self.gesture_frame_count += 1
                if self.gesture_frame_count >= self.gesture_hold_frames:
                    return True
            else:
                self.gesture_frame_count = 0
        except:
            self.gesture_frame_count = 0
        
        return False
    
    def voice_recognition_thread(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🎤 Listening for 'shadow clone jutsu'...\n")
        
        while self.listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: '{text}'")
                    
                    if "shadow clone jutsu" in text or "shadow clone" in text:
                        self.voice_queue.put("SHADOW_CLONE")
                        print("🔊 VOICE COMMAND DETECTED!")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                pass
    
    def extract_person(self, frame, pose_result):
        h, w, c = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if not pose_result.pose_landmarks:
            return frame, mask
        
        landmarks = pose_result.pose_landmarks[0]
        points = []
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        
        points = np.array(points)
        hull = cv2.convexHull(points)
        center = hull.mean(axis=0)
        expanded_hull = center + (hull - center) * 1.3
        
        cv2.fillConvexPoly(mask, expanded_hull.astype(np.int32), 255)
        person = cv2.bitwise_and(frame, frame, mask=mask)
        
        return person, mask
    
    def create_clone_effect(self, frame, person, mask):
        h, w, c = frame.shape
        output = frame.copy()
        
        # Shift clone to the right
        shift_x = w // 3
        M = np.float32([[1, 0, shift_x], [0, 1, 0]])
        clone_shifted = cv2.warpAffine(person, M, (w, h))
        mask_shifted = cv2.warpAffine(mask, M, (w, h))
        
        # Add tint to clone
        clone_tinted = clone_shifted.copy()
        clone_tinted[:, :, 0] = clone_tinted[:, :, 0] * 0.7
        clone_tinted[:, :, 1] = clone_tinted[:, :, 1] * 0.7
        
        # Blend
        mask_inv = cv2.bitwise_not(mask_shifted)
        background = cv2.bitwise_and(output, output, mask=mask_inv)
        clone_fg = cv2.bitwise_and(clone_tinted, clone_tinted, mask=mask_shifted)
        output = cv2.add(background, clone_fg)
        
        return output
    
    def run(self):
        if self.gesture_model is None:
            return
        
        cap = cv2.VideoCapture(0)
        
        # Start voice thread
        self.listening = True
        voice_thread = threading.Thread(target=self.voice_recognition_thread, daemon=True)
        voice_thread.start()
        
        print("\n" + "=" * 60)
        print("🥷 SHADOW CLONE JUTSU - ACTIVATED!")
        print("=" * 60)
        print("\nHow to use:")
        print("1. Make the cross-hand sign")
        print("2. Say 'shadow clone jutsu'")
        print("3. BOTH must happen together!")
        print("4. Clone appears for 5 seconds")
        print("\nPress 'Q' to quit | 'R' to reset")
        print("-" * 60 + "\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert to MediaPipe image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands and pose
            hand_result = self.hand_detector.detect(mp_image)
            pose_result = self.pose_detector.detect(mp_image)
            
            # Check voice
            try:
                while not self.voice_queue.empty():
                    command = self.voice_queue.get_nowait()
                    if command == "SHADOW_CLONE":
                        self.voice_command_detected = True
            except queue.Empty:
                pass
            
            # Detect gesture
            current_gesture = self.detect_gesture(hand_result)
            if current_gesture:
                self.gesture_detected = True
            
            # Activate clone
            if self.gesture_detected and self.voice_command_detected and not self.clone_active:
                print("\n💥 SHADOW CLONE JUTSU ACTIVATED!\n")
                
                person, mask = self.extract_person(frame, pose_result)
                self.clone_image = person
                self.clone_mask = mask
                self.clone_active = True
                self.clone_start_time = datetime.now()
                
                self.gesture_detected = False
                self.voice_command_detected = False
            
            # Show clone
            if self.clone_active:
                elapsed = (datetime.now() - self.clone_start_time).total_seconds()
                
                if elapsed < self.clone_duration:
                    frame = self.create_clone_effect(frame, self.clone_image, self.clone_mask)
                    remaining = self.clone_duration - elapsed
                    cv2.putText(frame, f"Clone: {remaining:.1f}s", 
                               (w//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 3)
                else:
                    self.clone_active = False
                    print("Clone disappeared\n")
            
            # Status UI
            status_bg = np.zeros((150, w, 3), dtype=np.uint8)
            
            gesture_color = (0, 255, 0) if current_gesture else (0, 0, 255)
            voice_color = (0, 255, 0) if self.voice_command_detected else (0, 0, 255)
            
            cv2.putText(status_bg, f"Gesture: {'DETECTED' if current_gesture else 'WAITING'}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
            cv2.putText(status_bg, f"Voice: {'DETECTED' if self.voice_command_detected else 'WAITING'}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, voice_color, 2)
            cv2.putText(status_bg, "Q = Quit | R = Reset", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            frame = np.vstack([status_bg, frame])
            
            cv2.imshow('Shadow Clone Jutsu', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.clone_active = False
                self.gesture_detected = False
                self.voice_command_detected = False
                print("Reset!\n")
        
        self.listening = False
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = ShadowCloneJutsu()
        app.run()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
