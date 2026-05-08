"""
Shadow Clone Jutsu — v10
────────────────────────────────────────────────────────────────
Hold the cross-sign to charge clones:
  3s=1  4s=2  5s=3  6s=4  7s+=5

SAGE MODE:
  Close both eyes for 3s → iris overlay activates
  Works in solo and clone mode
  S = reset sage mode

KEYS:  R=Reset  S=Reset Sage Mode  Q=Quit
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
    print("❌ Install: pip install mediapipe"); exit(1)

# ── Audio ───────────────────────────────────────────────────────
AUDIO_BACKEND = None
for _pkg in [("pygame", None), ("playsound", None)]:
    try:
        if _pkg[0] == "pygame":
            import pygame; pygame.mixer.init()
            AUDIO_BACKEND = "pygame"; print("✅ Audio: pygame"); break
        import playsound as _ps
        AUDIO_BACKEND = "playsound"; print("✅ Audio: playsound"); break
    except Exception:
        pass
if AUDIO_BACKEND is None:
    try:
        import subprocess
        subprocess.run(["ffplay","-version"],capture_output=True,check=True)
        AUDIO_BACKEND = "ffplay"; print("✅ Audio: ffplay")
    except Exception:
        print("⚠️  No audio backend.")

def play_audio(path):
    if not os.path.exists(path): return
    def _play():
        try:
            if AUDIO_BACKEND == "pygame":
                pygame.mixer.music.load(path); pygame.mixer.music.play()
            elif AUDIO_BACKEND == "playsound":
                _ps.playsound(path, block=True)
            elif AUDIO_BACKEND == "ffplay":
                import subprocess
                subprocess.run(["ffplay","-nodisp","-autoexit","-loglevel","quiet",path])
        except Exception as e:
            print(f"⚠️ Audio: {e}")
    threading.Thread(target=_play, daemon=True).start()

def held_to_clones(s):
    if s < 3: return 0
    if s < 4: return 1
    if s < 5: return 2
    if s < 6: return 3
    if s < 7: return 4
    return 5


# ══════════════════════════════════════════════════════════════
#  Sage Mode — iris overlay, activated by 3s eye-close
# ══════════════════════════════════════════════════════════════
class SageModeFilter:
    LEFT_IRIS  = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    # EAR landmarks (fallback)
    L_TOP=159; L_BOT=145; L_L=33;  L_R=133
    R_TOP=386; R_BOT=374; R_L=362; R_R=263
    EAR_THRESH   = 0.18
    BLINK_THRESH = 0.55   # blendshape score
    HOLD_SECS    = 3.0

    def __init__(self, overlay_path):
        self.overlay_img = None
        if os.path.exists(overlay_path):
            img = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
            if img is not None:
                self.overlay_img = img
                print(f"✅ Sage overlay: {overlay_path}")
            else:
                print(f"⚠️ Cannot read {overlay_path}")
        else:
            print(f"⚠️ Overlay not found: {overlay_path}")

        from mediapipe.tasks.python import vision as _v
        from mediapipe.tasks.python.core import base_options as _b
        _model = "face_landmarker.task"
        if not os.path.exists(_model):
            print(f"   Downloading {_model}…")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                _model)
        self.detector = _v.FaceLandmarker.create_from_options(
            _v.FaceLandmarkerOptions(
                base_options=_b.BaseOptions(model_asset_path=_model),
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5))
        print("✅ FaceLandmarker ready")

        self.active = False
        self._close_since = None
        self._last_result = None   # cached each frame

    # ── Run detection on raw frame, return result ──────────────
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)
        self._last_result = result
        return result

    # ── Check blink and maybe activate ────────────────────────
    def check_activation(self, result, now):
        if not result.face_landmarks:
            self._close_since = None
            return

        # prefer blendshapes (more reliable with FaceLandmarker)
        both_closed = False
        if result.face_blendshapes:
            bs = {b.category_name: b.score for b in result.face_blendshapes[0]}
            lb = bs.get("eyeBlinkLeft", 0.0)
            rb = bs.get("eyeBlinkRight", 0.0)
            both_closed = (lb > self.BLINK_THRESH and rb > self.BLINK_THRESH)
        else:
            lms = result.face_landmarks[0]
            h, w = 1, 1   # normalised — use raw x/y
            def ear(ti, bi, li, ri):
                v = abs(lms[ti].y - lms[bi].y)
                h2 = abs(lms[li].x - lms[ri].x) + 1e-6
                return v / h2
            both_closed = (ear(self.L_TOP,self.L_BOT,self.L_L,self.L_R) < self.EAR_THRESH and
                           ear(self.R_TOP,self.R_BOT,self.R_L,self.R_R) < self.EAR_THRESH)

        if both_closed:
            if self._close_since is None:
                self._close_since = now
            elif not self.active and (now - self._close_since) >= self.HOLD_SECS:
                self.active = True
                print("👁️  SAGE MODE ACTIVATED!")
        else:
            self._close_since = None

    # ── Draw countdown arc top-left ────────────────────────────
    def draw_ui(self, frame, now):
        if self._close_since is None or self.active:
            return
        elapsed = now - self._close_since
        pct = min(elapsed / self.HOLD_SECS, 1.0)
        remaining = max(0, self.HOLD_SECS - elapsed)
        cx, cy, r = 50, 50, 35
        cv2.circle(frame, (cx,cy), r, (20,20,20), -1)
        if int(pct*360) > 0:
            cv2.ellipse(frame,(cx,cy),(r-3,r-3),-90,0,int(pct*360),(0,220,200),5,cv2.LINE_AA)
        lbl = f"{remaining:.1f}s"
        (tw,_),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
        cv2.putText(frame,lbl,(cx-tw//2,cy+5),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,"SAGE",(cx-14,cy+r+16),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,220,200),1,cv2.LINE_AA)

    # ── Iris info from landmark list ───────────────────────────
    def _iris_info(self, lms, indices, w, h):
        cx = int(lms[indices[0]].x * w)
        cy = int(lms[indices[0]].y * h)
        rs = [np.hypot(lms[i].x*w - cx, lms[i].y*h - cy) for i in indices[1:]]
        return cx, cy, max(int(np.mean(rs)) if rs else 8, 3)

    # ── Paint overlay onto a BGR image in-place ────────────────
    def _paint_iris(self, img, lms, w, h):
        if self.overlay_img is None:
            return
        for idx_set in (self.LEFT_IRIS, self.RIGHT_IRIS):
            cx, cy, radius = self._iris_info(lms, idx_set, w, h)
            diam = radius * 2
            if diam < 4:
                continue
            ov = cv2.resize(self.overlay_img, (diam, diam), interpolation=cv2.INTER_AREA)
            circ = np.zeros((diam, diam), np.uint8)
            cv2.circle(circ, (radius, radius), radius, 255, -1)

            x1,y1 = cx-radius, cy-radius
            x2,y2 = x1+diam,   y1+diam
            fx1,fy1 = max(0,x1), max(0,y1)
            fx2,fy2 = min(w,x2), min(h,y2)
            ox1,oy1 = fx1-x1, fy1-y1
            ox2,oy2 = ox1+(fx2-fx1), oy1+(fy2-fy1)
            if fx2<=fx1 or fy2<=fy1: continue

            roi  = img[fy1:fy2, fx1:fx2].astype(np.float32)
            ov_c = ov[oy1:oy2, ox1:ox2].astype(np.float32)
            m    = circ[oy1:oy2, ox1:ox2].astype(np.float32)[:,:,None] / 255.0
            img[fy1:fy2, fx1:fx2] = np.clip(ov_c*m*0.9 + roi*(1-m*0.9), 0, 255).astype(np.uint8)

    # ── Apply to a full frame using last detection result ───────
    def apply_to_frame(self, frame, result=None):
        if not self.active or self.overlay_img is None:
            return frame
        r = result or self._last_result
        if r is None or not r.face_landmarks:
            return frame
        h, w = frame.shape[:2]
        self._paint_iris(frame, r.face_landmarks[0], w, h)
        return frame

    # ── Apply to a sprite using landmark coords mapped to sprite space
    def apply_to_sprite(self, sprite, result, orig_w, orig_h, crop_x, crop_y, scale):
        if not self.active or self.overlay_img is None:
            return sprite
        if result is None or not result.face_landmarks:
            return sprite
        lms = result.face_landmarks[0]
        sh, sw = sprite.shape[:2]

        # Build a fake landmark list mapped into sprite pixel coords
        class _FakeLM:
            def __init__(self, x, y): self.x=x; self.y=y

        def remap(idx):
            ox = (lms[idx].x * orig_w - crop_x) * scale
            oy = (lms[idx].y * orig_h - crop_y) * scale
            return _FakeLM(ox / sw, oy / sh)

        for idx_set in (self.LEFT_IRIS, self.RIGHT_IRIS):
            mapped = [remap(i) for i in idx_set]
            cx = int(mapped[0].x * sw)
            cy = int(mapped[0].y * sh)
            rs = [np.hypot((mapped[i].x-mapped[0].x)*sw,
                           (mapped[i].y-mapped[0].y)*sh) for i in range(1,5)]
            radius = max(int(np.mean(rs)) if rs else 4, 2)
            diam = radius * 2
            if diam < 4: continue
            ov = cv2.resize(self.overlay_img, (diam,diam), interpolation=cv2.INTER_AREA)
            circ = np.zeros((diam,diam),np.uint8)
            cv2.circle(circ,(radius,radius),radius,255,-1)
            x1,y1 = cx-radius,cy-radius
            x2,y2 = x1+diam, y1+diam
            fx1,fy1 = max(0,x1),max(0,y1)
            fx2,fy2 = min(sw,x2),min(sh,y2)
            ox1,oy1 = fx1-x1,fy1-y1
            ox2,oy2 = ox1+(fx2-fx1),oy1+(fy2-fy1)
            if fx2<=fx1 or fy2<=fy1: continue
            roi  = sprite[fy1:fy2,fx1:fx2].astype(np.float32)
            ov_c = ov[oy1:oy2,ox1:ox2].astype(np.float32)
            m    = circ[oy1:oy2,ox1:ox2].astype(np.float32)[:,:,None]/255.0
            sprite[fy1:fy2,fx1:fx2] = np.clip(ov_c*m*0.9+roi*(1-m*0.9),0,255).astype(np.uint8)
        return sprite

    def reset(self):
        self.active = False
        self._close_since = None
        print("👁️  Sage mode reset.")


# ══════════════════════════════════════════════════════════════
#  Main App
# ══════════════════════════════════════════════════════════════
class ShadowCloneApp:
    TINT       = (0.72, 0.72, 1.0)
    STATE_IDLE = "idle"
    STATE_CLN  = "clones"
    AUDIO_1    = "sc_audio.m4a"
    AUDIO_N    = "_multiple_sc_audio.m4a"
    THRESHOLDS = [3,4,5,6,7]

    # Clone scales — index = clone slot (0=first neighbour, etc.)
    CLONE_SCALES = [0.82, 0.72, 0.65, 0.60]

    def __init__(self):
        print("\n🥷 SHADOW CLONE JUTSU — v10\n")

        def _ensure(path, url):
            if not os.path.exists(path):
                print(f"   Downloading {path}…")
                urllib.request.urlretrieve(url, path)

        from mediapipe.tasks.python import vision as _v
        from mediapipe.tasks.python.core import base_options as _b

        _ensure("hand_landmarker.task",
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        self.hands = _v.HandLandmarker.create_from_options(
            _v.HandLandmarkerOptions(
                base_options=_b.BaseOptions(model_asset_path="hand_landmarker.task"),
                num_hands=2, min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))

        _ensure("pose_landmarker.task",
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        self.pose = _v.PoseLandmarker.create_from_options(
            _v.PoseLandmarkerOptions(
                base_options=_b.BaseOptions(model_asset_path="pose_landmarker.task"),
                min_pose_detection_confidence=0.5, min_tracking_confidence=0.5))

        if not os.path.exists("gesture_model.pkl"):
            print("❌ gesture_model.pkl not found!"); exit(1)
        with open("gesture_model.pkl","rb") as f:
            self.gmodel = pickle.load(f)

        self.sage = SageModeFilter("eye_overlay.jpg")

        # BG
        self.bg_model = None; self.bg_ready = False
        self.bg_cnt   = 0;    self.bg_need  = 30; self.bg_a = 0.02

        # State
        self.state   = self.STATE_IDLE
        self.nclones = 0
        self.bbox    = None   # current person bbox
        self.fw = 640; self.fh = 480
        self.CONF = 0.6

        # sign hold
        self.held_since = None; self.was_held = False

        # overlay msg
        self.ovmsg = ""; self.ovcol = (255,255,255); self.ovtil = 0.0

        # ── FROZEN clone data ──────────────────────────────────
        # Sprites are captured ONCE at activation, never re-extracted.
        # _clone_sprites: list of (sprite, mask, cx_on_canvas, base_bot)
        # We only update base_bot each frame from live pose Y.
        self._frozen_sprites  = []   # list of (sp, sm, crop_x, crop_y, scale)
        self._frozen_cx_list  = []   # list of target cx on canvas (fixed)
        self._frozen_bot      = None # Y bottom at activation (updated each frame from pose Y)
        self._frozen_pw       = 0    # person width at activation (for step calc)
        self._frozen_ph       = 0    # person height

        print("✅ READY  |  Hold cross-sign to charge")
        print("   Close eyes 3s → Sage Mode  |  S=Reset Sage  R=Reset  Q=Quit\n")

    # ── Background ────────────────────────────────────────────
    def update_bg(self, frame):
        f32 = frame.astype(np.float32)
        if self.bg_model is None: self.bg_model = f32.copy()
        if not self.bg_ready:
            self.bg_cnt += 1
            a = 1.0/self.bg_cnt
            self.bg_model = (1-a)*self.bg_model + a*f32
            if self.bg_cnt >= self.bg_need:
                self.bg_ready = True; print("✅ Background calibrated!")
        else:
            self.bg_model = (1-self.bg_a)*self.bg_model + self.bg_a*f32

    # ── Person mask ───────────────────────────────────────────
    def person_mask(self, frame, pose_res):
        h,w = frame.shape[:2]
        mask = np.zeros((h,w),np.uint8)
        if not pose_res.pose_landmarks:
            self.bbox = None; return mask
        lms  = pose_res.pose_landmarks[0]
        pts  = np.array([[int(lm.x*w),int(lm.y*h)] for lm in lms])
        hull = cv2.convexHull(pts)
        ctr  = hull.mean(axis=0)
        exp  = (ctr+(hull-ctr)*1.25).astype(np.int32)
        coarse = np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(coarse,exp,255)
        if self.bg_ready:
            bg  = np.clip(self.bg_model,0,255).astype(np.uint8)
            lf  = cv2.cvtColor(frame,cv2.COLOR_BGR2Lab)
            lb  = cv2.cvtColor(bg,   cv2.COLOR_BGR2Lab)
            diff= cv2.absdiff(lf,lb).astype(np.float32)
            dg  = (diff[:,:,0]*0.5+diff[:,:,1]*0.25+diff[:,:,2]*0.25).astype(np.uint8)
            _,fg= cv2.threshold(dg,18,255,cv2.THRESH_BINARY)
            k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            k9=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
            k21=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
            fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,k3)
            fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,k21)
            fg=cv2.morphologyEx(fg,cv2.MORPH_DILATE,k9)
            mask=cv2.bitwise_and(fg,coarse)
            mask=cv2.GaussianBlur(mask,(5,5),0)
            _,mask=cv2.threshold(mask,64,255,cv2.THRESH_BINARY)
        else:
            mask=coarse
        fl=mask.copy(); fm=np.zeros((h+2,w+2),np.uint8)
        cv2.floodFill(fl,fm,(0,0),255)
        mask=cv2.bitwise_or(mask,cv2.bitwise_not(fl))
        mask=cv2.bitwise_and(mask,coarse)
        coords=cv2.findNonZero(mask)
        self.bbox=tuple(cv2.boundingRect(coords)) if coords is not None else None
        return mask

    # ── Make one tinted+scaled sprite ─────────────────────────
    def _make_sprite(self, frame, mask, scale=1.0):
        coords=cv2.findNonZero(mask)
        if coords is None: return None,None,0,0
        x,y,bw,bh=cv2.boundingRect(coords)
        tinted=frame.copy().astype(np.float32)
        for i,t in enumerate(self.TINT): tinted[:,:,i]*=t
        tinted=np.clip(tinted,0,255).astype(np.uint8)
        crop =cv2.bitwise_and(tinted,tinted,mask=mask)[y:y+bh,x:x+bw]
        mcrop=mask[y:y+bh,x:x+bw]
        nw,nh=max(1,int(bw*scale)),max(1,int(bh*scale))
        sp=cv2.resize(crop, (nw,nh),interpolation=cv2.INTER_AREA)
        sm=cv2.resize(mcrop,(nw,nh),interpolation=cv2.INTER_AREA)
        _,sm=cv2.threshold(sm,64,255,cv2.THRESH_BINARY)
        return sp,sm,x,y   # also return crop origin for sage mapping

    # ── Composite sprite onto canvas ──────────────────────────
    def composite(self, canvas, sp, sm, cx, bot):
        ch,cw=canvas.shape[:2]; sh,sw=sp.shape[:2]
        x1=cx-sw//2; y1=bot-sh; x2=x1+sw; y2=y1+sh
        sx1=max(0,-x1); sy1=max(0,-y1)
        sx2=sw-max(0,x2-cw); sy2=sh-max(0,y2-ch)
        dx1=max(0,x1);dy1=max(0,y1);dx2=min(cw,x2);dy2=min(ch,y2)
        if dx2<=dx1 or dy2<=dy1 or sx2<=sx1 or sy2<=sy1: return canvas
        m  =sm[sy1:sy2,sx1:sx2]
        roi=canvas[dy1:dy2,dx1:dx2]
        bg =cv2.bitwise_and(roi,roi,mask=cv2.bitwise_not(m))
        fg =cv2.bitwise_and(sp[sy1:sy2,sx1:sx2],sp[sy1:sy2,sx1:sx2],mask=m)
        canvas[dy1:dy2,dx1:dx2]=cv2.add(bg,fg)
        return canvas

    # ── Freeze sprites at activation ──────────────────────────
    def _freeze_clones(self, frame, mask, n, face_result):
        """
        Capture sprite pixels RIGHT NOW and compute fixed canvas X positions.
        Called once when clones activate.
        """
        if self.bbox is None: return
        px,py,pw,ph = self.bbox
        pcx = px + pw//2
        pbot= py + ph

        self._frozen_sprites = []
        self._frozen_cx_list = []
        self._frozen_bot     = pbot
        self._frozen_pw      = pw
        self._frozen_ph      = ph

        step = max(int(pw * 1.1), 80)

        if n == 1:
            # 1 clone: mirrored on the opposite side, slightly smaller
            sp,sm,cx0,cy0 = self._make_sprite(frame, mask, scale=0.85)
            if sp is not None:
                if self.sage.active and face_result:
                    sp = self.sage.apply_to_sprite(sp, face_result,
                                                   self.fw, self.fh, cx0, cy0, 0.85)
                clone_cx = self.fw - pcx
                self._frozen_sprites.append((sp, sm))
                self._frozen_cx_list.append(clone_cx)
        else:
            # n-1 clones around frozen centre, scaling down with distance
            offsets = []
            for i in range(1, n):
                side = i % 2
                dist = (i+1)//2
                offsets.append(dist * step * (1 if side else -1))

            for idx, ox in enumerate(offsets):
                sc = self.CLONE_SCALES[min(idx, len(self.CLONE_SCALES)-1)]
                sp,sm,cx0,cy0 = self._make_sprite(frame, mask, scale=sc)
                if sp is not None:
                    if self.sage.active and face_result:
                        sp = self.sage.apply_to_sprite(sp, face_result,
                                                       self.fw, self.fh, cx0, cy0, sc)
                    self._frozen_sprites.append((sp, sm))
                    self._frozen_cx_list.append(pcx + ox)

        print(f"   Frozen {len(self._frozen_sprites)} clone sprite(s) at cx={pcx}, bot={pbot}")

    # ── Render frozen clones (YOU redrawn live on top) ────────
    def render_clones(self, frame, mask, face_result):
        canvas = frame.copy()

        # Live Y bottom from current pose
        if self.bbox:
            _,py,_,ph = self.bbox
            live_bot = py + ph
        else:
            live_bot = self._frozen_bot or frame.shape[0]

        # Draw frozen clones at their fixed X, updated Y
        for (sp, sm), cx in zip(self._frozen_sprites, self._frozen_cx_list):
            canvas = self.composite(canvas, sp, sm, cx, live_bot)

        # YOU: re-extracted live so you move freely, drawn on top
        if self.bbox:
            px,py,pw,ph = self.bbox
            pcx_live = px+pw//2
            pbot_live= py+ph
            sp_you,sm_you,cx0,cy0 = self._make_sprite(frame, mask, scale=1.0)
            if sp_you is not None:
                if self.sage.active and face_result:
                    sp_you = self.sage.apply_to_sprite(sp_you, face_result,
                                                       self.fw, self.fh, cx0, cy0, 1.0)
                canvas = self.composite(canvas, sp_you, sm_you, pcx_live, pbot_live)

        return canvas

    # ── Gesture ───────────────────────────────────────────────
    def detect_gesture(self, result):
        if not result.hand_landmarks or self.gmodel is None: return False
        feats=[v for lms in result.hand_landmarks for lm in lms for v in (lm.x,lm.y,lm.z)]
        try:
            arr=np.array(feats).reshape(1,-1)
            pred=self.gmodel.predict(arr)[0]
            prob=self.gmodel.predict_proba(arr)[0]
            return pred==1 and prob[1]>self.CONF
        except: return False

    # ── Activate ──────────────────────────────────────────────
    def activate(self, n, frame, mask, face_result):
        self.nclones = n
        self._freeze_clones(frame, mask, n, face_result)
        if n==1:
            play_audio(self.AUDIO_1)
            self.set_ov(f"💥  SHADOW CLONE JUTSU!  [{n} clone]",(0,255,120))
        else:
            play_audio(self.AUDIO_N)
            self.set_ov(f"💥  MULTIPLE SHADOW CLONE JUTSU!  [{n} clones]",(255,200,0))
        print(f"\n💥 {n} CLONE(S) ACTIVATED!\n")
        self.state = self.STATE_CLN

    def set_ov(self,msg,col=(255,255,255),dur=3.0):
        self.ovmsg=msg; self.ovcol=col; self.ovtil=time.time()+dur

    # ── Charge timer ──────────────────────────────────────────
    def draw_charge(self, frame, sec):
        fw=frame.shape[1]; cx,cy=fw-80,80; r=55
        cv2.circle(frame,(cx,cy),r,(30,30,30),-1)
        cv2.circle(frame,(cx,cy),r,(60,60,60),2)
        pn=held_to_clones(sec)
        cols=[(80,80,80),(0,255,120),(0,255,220),(0,200,255),(0,140,255),(0,80,255)]
        arc_col=cols[min(pn,5)] if sec>=3 else cols[0]
        deg=int(min(sec/7,1)*360)
        if deg>0:
            cv2.ellipse(frame,(cx,cy),(r-4,r-4),-90,0,deg,arc_col,6,cv2.LINE_AA)
        for thr in self.THRESHOLDS:
            ang=((thr/7)*360-90)*np.pi/180
            cv2.line(frame,(int(cx+(r-2)*np.cos(ang)),int(cy+(r-2)*np.sin(ang))),
                           (int(cx+(r+6)*np.cos(ang)),int(cy+(r+6)*np.sin(ang))),(200,200,200),2,cv2.LINE_AA)
        s=f"{sec:.1f}s"
        (tw,_),_=cv2.getTextSize(s,cv2.FONT_HERSHEY_DUPLEX,0.55,1)
        cv2.putText(frame,s,(cx-tw//2,cy+5),cv2.FONT_HERSHEY_DUPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
        badge=f"x{pn}" if pn>0 else "hold..."
        (bw,_),_=cv2.getTextSize(badge,cv2.FONT_HERSHEY_DUPLEX,0.6,2)
        cv2.putText(frame,badge,(cx-bw//2,cy+r+22),cv2.FONT_HERSHEY_DUPLEX,0.6,arc_col if pn>0 else (100,100,100),2,cv2.LINE_AA)
        if pn>0:
            pulse=int(160+95*abs(np.sin(sec*4)))
            cv2.circle(frame,(cx,cy),r+10,(*arc_col[:2],pulse),2,cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────
    def draw_hud(self, frame, held):
        fh,fw=frame.shape[:2]
        hud=np.zeros((48,fw,3),np.uint8)
        sage_tag=" | 👁 SAGE" if self.sage.active else ""
        if self.state==self.STATE_IDLE:
            pn=held_to_clones(held) if held else 0
            msg=(f"Holding {held:.1f}s → {'Release for '+str(pn)+' clone(s)!' if pn else 'Keep holding…'}"
                 if held else f"Hold cross-sign to charge | 3s=1 4s=2 5s=3 6s=4 7s=5{sage_tag}")
            col=(0,255,120) if pn else ((0,220,200) if self.sage.active else (140,140,140))
        else:
            msg=f"{self.nclones} CLONE{'S' if self.nclones>1 else ''}  |  R=Reset  Q=Quit{sage_tag}"
            col=(0,220,200) if self.sage.active else (80,80,80)
        cv2.putText(hud,msg,(16,30),cv2.FONT_HERSHEY_SIMPLEX,0.46,col,1,cv2.LINE_AA)
        return np.vstack([frame,hud])

    # ── Main loop ─────────────────────────────────────────────
    def run(self):
        cap=cv2.VideoCapture(0)
        for rw,rh in [(1280,720),(960,540),(854,480),(640,480)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,rw); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,rh)
            if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))>=rw-10: break
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"MJPG"))
        self.fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📷  Camera: {self.fw}×{self.fh}")

        # Keep a mask for the activation frame (for _freeze_clones)
        _last_mask = None

        while cap.isOpened():
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.flip(frame,1)
            fh,fw=frame.shape[:2]
            self.fw=fw; self.fh=fh
            now=time.time()

            # ── Face detection on the RAW frame first ──────────
            # (before any compositing, so iris detection is clean)
            face_result = self.sage.detect(frame)
            self.sage.check_activation(face_result, now)

            # ── Pose ──────────────────────────────────────────
            rgb   =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            pose_res=self.pose.detect(mp_img)

            if self.state!=self.STATE_CLN:
                self.update_bg(frame)

            held=0.0

            if self.state==self.STATE_IDLE:
                mask=self.person_mask(frame,pose_res)
                _last_mask=mask
                hand_res=self.hands.detect(mp_img)
                sign=self.detect_gesture(hand_res)

                if sign:
                    if self.held_since is None:
                        self.held_since=now; self.was_held=True
                        print("✋ Charging…")
                    held=min(now-self.held_since,7.5)
                    self.draw_charge(frame,held)
                else:
                    if self.was_held and self.held_since:
                        hf=now-self.held_since; n=held_to_clones(hf)
                        if n>0:
                            self.activate(n, frame, mask, face_result)
                        else:
                            self.set_ov("Hold longer! (3s min)",(0,160,255),2.0)
                            print(f"   Too early ({hf:.1f}s)")
                    self.held_since=None; self.was_held=False; held=0.0

            elif self.state==self.STATE_CLN:
                mask=self.person_mask(frame,pose_res)
                frame=self.render_clones(frame, mask, face_result)

            # ── Sage filter on YOU (live frame, after clone render) ─
            frame = self.sage.apply_to_frame(frame, face_result)
            self.sage.draw_ui(frame, now)

            if self.sage.active:
                cv2.putText(frame,"SAGE MODE",(fw-135,fh-10),
                            cv2.FONT_HERSHEY_DUPLEX,0.6,(0,220,200),2,cv2.LINE_AA)

            if not self.bg_ready and self.state!=self.STATE_CLN:
                pct=int(self.bg_cnt/self.bg_need*100)
                cv2.putText(frame,f"Calibrating BG… {pct}%",(10,fh-20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,200,255),1,cv2.LINE_AA)

            frame=self.draw_hud(frame,held if self.state==self.STATE_IDLE else 0)

            if self.ovmsg and now<self.ovtil:
                font,sc,tk=cv2.FONT_HERSHEY_DUPLEX,0.9,2
                (tw,_),_=cv2.getTextSize(self.ovmsg,font,sc,tk)
                tx=(fw-tw)//2
                cv2.putText(frame,self.ovmsg,(tx+2,62),font,sc,(0,0,0),tk+2,cv2.LINE_AA)
                cv2.putText(frame,self.ovmsg,(tx,60),font,sc,self.ovcol,tk,cv2.LINE_AA)

            cv2.imshow("Shadow Clone Jutsu",frame)

            key=cv2.waitKey(1)&0xFF
            if key==ord("q"): break
            elif key==ord("r"):
                self.state=self.STATE_IDLE; self.nclones=0
                self.held_since=None; self.was_held=False
                self.bg_model=None; self.bg_ready=False; self.bg_cnt=0
                self.bbox=None
                self._frozen_sprites=[]; self._frozen_cx_list=[]
                self._frozen_bot=None
                self.set_ov("Reset!",(180,180,180),1.5)
                print("\n🔄 Reset\n")
            elif key==ord("s"):
                self.sage.reset()
                self.set_ov("Sage Mode Reset",(0,220,200),1.5)

        cap.release(); cv2.destroyAllWindows()


if __name__=="__main__":
    try:
        app=ShadowCloneApp(); app.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        import traceback; print(f"\n❌ {e}"); traceback.print_exc()