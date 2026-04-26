"""
Shadow Clone Jutsu — v8
────────────────────────────────────────────────────────────────
NO VOICE. Hold the cross-sign — timer shown on screen.
Release to spawn clones based on how long you held:

  Hold 3s  → 1 clone
  Hold 4s  → 2 clones
  Hold 5s  → 3 clones
  Hold 6s  → 4 clones
  Hold 6s+ → 5 clones (max)

  Drop before 3s → resets, nothing happens.

KEYS:  R=Reset  Q=Quit
"""

import cv2
import numpy as np
import pickle
import os
import urllib.request
import time
import threading

print("Loading dependencies...")
try:
    import mediapipe as mp
    print(f"✅ MediaPipe loaded (v{mp.__version__})")
except ImportError:
    print("❌ Install: pip install mediapipe")
    exit(1)

# ── Audio playback ──────────────────────────────────────────────
AUDIO_BACKEND = None
try:
    import pygame
    pygame.mixer.init()
    AUDIO_BACKEND = "pygame"
    print("✅ Audio: pygame")
except Exception:
    pass

if AUDIO_BACKEND is None:
    try:
        import playsound as _ps
        AUDIO_BACKEND = "playsound"
        print("✅ Audio: playsound")
    except ImportError:
        pass

if AUDIO_BACKEND is None:
    try:
        import subprocess
        subprocess.run(["ffplay", "-version"], capture_output=True, check=True)
        AUDIO_BACKEND = "ffplay"
        print("✅ Audio: ffplay")
    except Exception:
        print("⚠️  No audio backend. Install: pip install pygame")


def play_audio(path: str):
    if not os.path.exists(path):
        print(f"⚠️  Audio file not found: {path}")
        return
    def _play():
        try:
            if AUDIO_BACKEND == "pygame":
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
            elif AUDIO_BACKEND == "playsound":
                _ps.playsound(path, block=True)
            elif AUDIO_BACKEND == "ffplay":
                import subprocess
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                    check=False)
        except Exception as e:
            print(f"⚠️  Audio error: {e}")
    threading.Thread(target=_play, daemon=True).start()


# ── Clone count from hold duration ─────────────────────────────
def held_to_clones(seconds: float) -> int:
    """Map hold duration → number of clones (0 if < 3s)."""
    if seconds < 3.0: return 0
    if seconds < 4.0: return 1
    if seconds < 5.0: return 2
    if seconds < 6.0: return 3
    if seconds < 7.0: return 4
    return 5  # 7s+ → max


# ──────────────────────────────────────────────────────────────
#  Main Application
# ──────────────────────────────────────────────────────────────
class ShadowCloneApp:

    TINT = (0.72, 0.72, 1.0)   # bluish clone tint

    STATE_IDLE   = "idle"    # waiting for cross-sign
    STATE_CLONES = "clones"  # clones active

    AUDIO_SINGLE   = "sc_audio.m4a"
    AUDIO_MULTIPLE = "_multiple_sc_audio.m4a"

    # Thresholds for ring colour feedback
    THRESHOLDS = [3, 4, 5, 6, 7]  # seconds

    def __init__(self):
        print("\n🥷 SHADOW CLONE JUTSU — v8\n")

        # ── MediaPipe hand detector ───────────────────────────
        self.hand_model_path = "hand_landmarker.task"
        self._ensure_file(
            self.hand_model_path,
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as bo
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=bo.BaseOptions(model_asset_path=self.hand_model_path),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5))
        print("✅ Hand detector ready")

        # ── MediaPipe pose detector ───────────────────────────
        self.pose_model_path = "pose_landmarker.task"
        self._ensure_file(
            self.pose_model_path,
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        from mediapipe.tasks.python import vision as vis2
        from mediapipe.tasks.python.core import base_options as bo2
        self.pose_detector = vis2.PoseLandmarker.create_from_options(
            vis2.PoseLandmarkerOptions(
                base_options=bo2.BaseOptions(model_asset_path=self.pose_model_path),
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5))
        print("✅ Pose detector ready")

        # ── Gesture model ─────────────────────────────────────
        if not os.path.exists("gesture_model.pkl"):
            print("❌ gesture_model.pkl not found!")
            exit(1)
        with open("gesture_model.pkl", "rb") as f:
            self.gesture_model = pickle.load(f)
        print("✅ Gesture model loaded")

        # ── Background subtraction ────────────────────────────
        self.bg_model         = None
        self.bg_ready         = False
        self.bg_frame_count   = 0
        self.bg_frames_needed = 30
        self.bg_alpha         = 0.02

        # ── App state ─────────────────────────────────────────
        self.state       = self.STATE_IDLE
        self.num_clones  = 0
        self.person_bbox = None
        self.fw = 640; self.fh = 480
        self.gesture_confidence_threshold = 0.6

        # sign hold tracking
        self.sign_held_since = None   # time.time() when sign first detected
        self.sign_was_held   = False  # True while sign is being held

        # overlay
        self.overlay_msg   = ""
        self.overlay_color = (255, 255, 255)
        self.overlay_until = 0.0

        for f in (self.AUDIO_SINGLE, self.AUDIO_MULTIPLE):
            print(f"{'✅' if os.path.exists(f) else '⚠️ '} Audio: {f}")

        print("\n" + "="*60)
        print("✅ READY — Hold the cross-sign to charge clones!")
        print("   3s=1 clone  4s=2  5s=3  6s=4  7s+=5")
        print("   R=Reset  Q=Quit")
        print("="*60 + "\n")

    # ── utils ─────────────────────────────────────────────────
    def _ensure_file(self, path, url):
        if not os.path.exists(path):
            print(f"   Downloading {path}...")
            urllib.request.urlretrieve(url, path)

    def set_overlay(self, msg, color=(255,255,255), duration=3.0):
        self.overlay_msg   = msg
        self.overlay_color = color
        self.overlay_until = time.time() + duration

    # ── background ────────────────────────────────────────────
    def update_background(self, frame):
        f32 = frame.astype(np.float32)
        if self.bg_model is None:
            self.bg_model = f32.copy()
        if not self.bg_ready:
            self.bg_frame_count += 1
            alpha = 1.0 / self.bg_frame_count
            self.bg_model = (1-alpha)*self.bg_model + alpha*f32
            if self.bg_frame_count >= self.bg_frames_needed:
                self.bg_ready = True
                print("✅ Background calibrated!")
        else:
            self.bg_model = (1-self.bg_alpha)*self.bg_model + self.bg_alpha*f32

    # ── person mask ───────────────────────────────────────────
    def extract_clean_person(self, frame, pose_result):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if not pose_result.pose_landmarks:
            self.person_bbox = None
            return mask
        lms  = pose_result.pose_landmarks[0]
        pts  = np.array([[int(lm.x*w), int(lm.y*h)] for lm in lms])
        hull = cv2.convexHull(pts)
        ctr  = hull.mean(axis=0)
        exp  = (ctr + (hull - ctr)*1.25).astype(np.int32)
        coarse = np.zeros((h,w), dtype=np.uint8)
        cv2.fillConvexPoly(coarse, exp, 255)
        if self.bg_ready:
            bg   = np.clip(self.bg_model,0,255).astype(np.uint8)
            lf   = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            lb   = cv2.cvtColor(bg,    cv2.COLOR_BGR2Lab)
            diff = cv2.absdiff(lf, lb).astype(np.float32)
            dg   = (diff[:,:,0]*0.5+diff[:,:,1]*0.25+diff[:,:,2]*0.25).astype(np.uint8)
            _, fg = cv2.threshold(dg, 18, 255, cv2.THRESH_BINARY)
            k3   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            k9   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
            k21  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
            fg   = cv2.morphologyEx(fg, cv2.MORPH_OPEN,   k3)
            fg   = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,  k21)
            fg   = cv2.morphologyEx(fg, cv2.MORPH_DILATE, k9)
            mask = cv2.bitwise_and(fg, coarse)
            mask = cv2.GaussianBlur(mask,(5,5),0)
            _,mask = cv2.threshold(mask,64,255,cv2.THRESH_BINARY)
        else:
            mask = coarse
        fl = mask.copy()
        fm = np.zeros((h+2,w+2),dtype=np.uint8)
        cv2.floodFill(fl,fm,(0,0),255)
        mask = cv2.bitwise_or(mask, cv2.bitwise_not(fl))
        mask = cv2.bitwise_and(mask, coarse)
        coords = cv2.findNonZero(mask)
        self.person_bbox = tuple(cv2.boundingRect(coords)) if coords is not None else None
        return mask

    # ── sprites ───────────────────────────────────────────────
    def make_clone_sprite(self, frame, mask, scale=1.0):
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None, None
        x,y,bw,bh = cv2.boundingRect(coords)
        tinted = frame.copy().astype(np.float32)
        for i,t in enumerate(self.TINT):
            tinted[:,:,i] *= t
        tinted = np.clip(tinted,0,255).astype(np.uint8)
        crop  = cv2.bitwise_and(tinted,tinted,mask=mask)[y:y+bh,x:x+bw]
        mcrop = mask[y:y+bh,x:x+bw]
        nw,nh = max(1,int(bw*scale)), max(1,int(bh*scale))
        sp = cv2.resize(crop,  (nw,nh), interpolation=cv2.INTER_AREA)
        sm = cv2.resize(mcrop, (nw,nh), interpolation=cv2.INTER_AREA)
        _,sm = cv2.threshold(sm,64,255,cv2.THRESH_BINARY)
        return sp, sm

    def composite(self, canvas, sprite, smask, cx, bottom_y):
        ch,cw = canvas.shape[:2]
        sh,sw = sprite.shape[:2]
        x1=cx-sw//2; y1=bottom_y-sh; x2=x1+sw; y2=y1+sh
        sx1=max(0,-x1); sy1=max(0,-y1)
        sx2=sw-max(0,x2-cw); sy2=sh-max(0,y2-ch)
        dx1=max(0,x1); dy1=max(0,y1); dx2=min(cw,x2); dy2=min(ch,y2)
        if dx2<=dx1 or dy2<=dy1 or sx2<=sx1 or sy2<=sy1:
            return canvas
        m   = smask[sy1:sy2,sx1:sx2]
        roi = canvas[dy1:dy2,dx1:dx2]
        bg  = cv2.bitwise_and(roi,roi,mask=cv2.bitwise_not(m))
        fg  = cv2.bitwise_and(sprite[sy1:sy2,sx1:sx2],sprite[sy1:sy2,sx1:sx2],mask=m)
        canvas[dy1:dy2,dx1:dx2] = cv2.add(bg,fg)
        return canvas

    # ── render clones ─────────────────────────────────────────
    def render_clones(self, frame, mask, n):
        """Render n clones (1–5). YOU always tinted and on top."""
        if self.person_bbox is None: return frame
        px,py,pw,ph = self.person_bbox
        pcx  = px + pw//2
        pbot = py + ph

        canvas = frame.copy()

        if n == 1:
            # one clone mirrored opposite side
            sp,sm = self.make_clone_sprite(frame, mask, scale=1.0)
            if sp is not None:
                clone_cx = self.fw - pcx
                canvas = self.composite(canvas, sp, sm, clone_cx, pbot)

        else:
            # n-1 clones spread around you (scaled down), you in centre
            scales = [0.60, 0.70, 0.78, 0.85]  # for clones 2-5
            # positions: alternate left/right of you
            offsets = []
            step = max(int(pw * 1.05), 70)
            # build left/right pairs: (left1, right1, left2, right2 ...)
            for i in range(1, n):
                side = i % 2   # 1=right, 0=left
                dist = (i + 1) // 2
                ox = dist * step * (1 if side else -1)
                offsets.append(ox)

            for idx, ox in enumerate(offsets):
                sc = scales[min(idx, len(scales)-1)]
                sp,sm = self.make_clone_sprite(frame, mask, scale=sc)
                if sp is not None:
                    # back clones raised slightly
                    by = pbot - int(ph * 0.05 * (idx // 2))
                    canvas = self.composite(canvas, sp, sm, pcx + ox, by)

        # YOU on top, also tinted
        sp_you, sm_you = self.make_clone_sprite(frame, mask, scale=1.0)
        if sp_you is not None:
            canvas = self.composite(canvas, sp_you, sm_you, pcx, pbot)

        return canvas

    # ── gesture ───────────────────────────────────────────────
    def detect_gesture(self, result):
        if not result.hand_landmarks or self.gesture_model is None:
            return False
        feats = [v for lms in result.hand_landmarks for lm in lms
                   for v in (lm.x, lm.y, lm.z)]
        try:
            arr  = np.array(feats).reshape(1,-1)
            pred = self.gesture_model.predict(arr)[0]
            prob = self.gesture_model.predict_proba(arr)[0]
            return pred==1 and prob[1] > self.gesture_confidence_threshold
        except Exception:
            return False

    # ── activation ────────────────────────────────────────────
    def activate(self, n):
        self.num_clones = n
        if n == 1:
            play_audio(self.AUDIO_SINGLE)
            self.set_overlay(f"💥  SHADOW CLONE JUTSU!  [{n} clone]", (0,255,120), 3.0)
        else:
            play_audio(self.AUDIO_MULTIPLE)
            self.set_overlay(f"💥  MULTIPLE SHADOW CLONE JUTSU!  [{n} clones]", (255,200,0), 3.0)
        print(f"\n💥 {n} CLONE(S) ACTIVATED!\n")
        self.state = self.STATE_CLONES

    # ── charge timer UI ───────────────────────────────────────
    def draw_charge_timer(self, frame, held_sec, now):
        """Draw a big arc timer + clone preview count on the frame."""
        fw = frame.shape[1]
        fh = frame.shape[0]
        cx, cy = fw - 80, 80   # top-right area
        radius = 55

        # Background circle
        cv2.circle(frame, (cx, cy), radius, (30,30,30), -1)
        cv2.circle(frame, (cx, cy), radius, (60,60,60), 2)

        # How many clones would pop if released now
        preview_n = held_to_clones(held_sec)

        # Arc fill — full arc = 7s
        arc_pct  = min(held_sec / 7.0, 1.0)
        arc_deg  = int(arc_pct * 360)

        # Colour: grey before 3s, then green→yellow→orange→red as clones go up
        if held_sec < 3.0:
            arc_col = (80, 80, 80)
        elif preview_n == 1:
            arc_col = (0, 255, 120)
        elif preview_n == 2:
            arc_col = (0, 255, 220)
        elif preview_n == 3:
            arc_col = (0, 200, 255)
        elif preview_n == 4:
            arc_col = (0, 140, 255)
        else:
            arc_col = (0, 80, 255)

        # Draw arc as ellipse (OpenCV ellipse uses degrees from 3-o'clock, CCW)
        if arc_deg > 0:
            cv2.ellipse(frame, (cx,cy), (radius-4, radius-4),
                        -90, 0, arc_deg, arc_col, 6, cv2.LINE_AA)

        # Threshold tick marks at 3,4,5,6,7s
        for thr in self.THRESHOLDS:
            ang = ((thr / 7.0) * 360 - 90) * np.pi / 180
            ix = int(cx + (radius-2) * np.cos(ang))
            iy = int(cy + (radius-2) * np.sin(ang))
            ox = int(cx + (radius+6) * np.cos(ang))
            oy = int(cy + (radius+6) * np.sin(ang))
            cv2.line(frame, (ix,iy), (ox,oy), (200,200,200), 2, cv2.LINE_AA)

        # Centre: seconds held
        sec_str = f"{held_sec:.1f}s"
        (tw,th),_ = cv2.getTextSize(sec_str, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.putText(frame, sec_str, (cx-tw//2, cy+5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        # Clone count badge below circle
        if preview_n > 0:
            badge = f"x{preview_n}"
            col   = arc_col
        else:
            badge = "hold..."
            col   = (100, 100, 100)
        (bw,_),_ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        cv2.putText(frame, badge, (cx-bw//2, cy+radius+22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2, cv2.LINE_AA)

        # Pulsing outer ring when locked onto a clone count
        if preview_n > 0:
            pulse = int(160 + 95 * abs(np.sin(held_sec * 4)))
            cv2.circle(frame, (cx,cy), radius+10, (*arc_col[:2], pulse), 2, cv2.LINE_AA)

    # ── HUD bar ───────────────────────────────────────────────
    def draw_hud(self, frame, now, held_sec):
        fh, fw = frame.shape[:2]
        hud = np.zeros((48, fw, 3), dtype=np.uint8)

        if self.state == self.STATE_IDLE:
            preview_n = held_to_clones(held_sec) if held_sec else 0
            if held_sec and held_sec > 0:
                msg = f"Holding {held_sec:.1f}s  →  {'Release for ' + str(preview_n) + ' clone(s)!' if preview_n > 0 else 'Keep holding... (3s=1 clone)'}"
                col = (0,255,120) if preview_n > 0 else (0,180,255)
            else:
                msg = "Hold the cross-sign to charge  |  3s=1  4s=2  5s=3  6s=4  7s=5"
                col = (140,140,140)
            cv2.putText(hud, msg, (16,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)

        else:  # clones active
            lbl = f"{self.num_clones} CLONE{'S' if self.num_clones>1 else ''}  |  R=Reset  Q=Quit"
            cv2.putText(hud, lbl, (16,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80,80,80), 1, cv2.LINE_AA)

        return np.vstack([frame, hud])

    # ── main loop ─────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)
        for rw,rh in [(1280,720),(960,540),(854,480),(640,480)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  rw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
            if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) >= rw-10:
                break
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📷  Camera: {self.fw}×{self.fh}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            self.fw = fw
            now = time.time()

            if self.state != self.STATE_CLONES:
                self.update_background(frame)

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose_result = self.pose_detector.detect(mp_img)

            # ── STATE MACHINE ──────────────────────────────
            held_sec = 0.0

            if self.state == self.STATE_IDLE:
                self.extract_clean_person(frame, pose_result)
                hand_result = self.hand_detector.detect(mp_img)
                sign_on = self.detect_gesture(hand_result)

                if sign_on:
                    if self.sign_held_since is None:
                        self.sign_held_since = now
                        self.sign_was_held = True
                        print("✋ Cross-sign held — charging...")
                    held_sec = now - self.sign_held_since

                    # Draw the charge timer arc on frame
                    self.draw_charge_timer(frame, held_sec, now)

                    # Cap at 5 clones
                    held_sec = min(held_sec, 7.5)

                else:
                    # sign dropped
                    if self.sign_was_held and self.sign_held_since is not None:
                        held_sec_final = now - self.sign_held_since
                        n = held_to_clones(held_sec_final)
                        if n > 0:
                            self.activate(n)
                        else:
                            print(f"   Released too early ({held_sec_final:.1f}s) — need 3s")
                            self.set_overlay("Hold longer! (3s minimum)", (0,160,255), 2.0)
                    self.sign_held_since = None
                    self.sign_was_held   = False
                    held_sec = 0.0

            elif self.state == self.STATE_CLONES:
                mask = self.extract_clean_person(frame, pose_result)
                frame = self.render_clones(frame, mask, self.num_clones)

            # BG calibration progress
            if not self.bg_ready and self.state != self.STATE_CLONES:
                pct = int(self.bg_frame_count/self.bg_frames_needed*100)
                cv2.putText(frame, f"Calibrating BG... {pct}%",
                            (10,fh-20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (0,200,255), 1, cv2.LINE_AA)

            # HUD
            frame = self.draw_hud(frame, now, held_sec if self.state == self.STATE_IDLE else 0)

            # Overlay
            if self.overlay_msg and now < self.overlay_until:
                msg = self.overlay_msg
                font,sc,tk = cv2.FONT_HERSHEY_DUPLEX, 0.9, 2
                (tw,_),_ = cv2.getTextSize(msg,font,sc,tk)
                tx = (fw-tw)//2
                cv2.putText(frame,msg,(tx+2,62),font,sc,(0,0,0),tk+2,cv2.LINE_AA)
                cv2.putText(frame,msg,(tx,   60),font,sc,self.overlay_color,tk,cv2.LINE_AA)

            cv2.imshow("Shadow Clone Jutsu", frame)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.state           = self.STATE_IDLE
                self.num_clones      = 0
                self.sign_held_since = None
                self.sign_was_held   = False
                self.bg_model        = None
                self.bg_ready        = False
                self.bg_frame_count  = 0
                self.person_bbox     = None
                self.set_overlay("Reset!", (180,180,180), 1.5)
                print("\n🔄 Reset\n")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app = ShadowCloneApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
