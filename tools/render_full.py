"""The full 10 s loop, picture only. Everything is computed from one timetable.

Usage: python render_full.py <project_dir> [--check-only] [--dump=20,27,150] [--half=NAME --closed=NAME] [--out=NAME]

Reads  assets/still.png, assets/blink_half.png, assets/blink_closed.png (full frames, eyes changed only), masks/*
Writes out/final_picture_master.mp4 (2048x1536), out/final_picture_1440.mp4, out/step4_checks.json  (--out=NAME changes the name)
"""
import sys, os, json, subprocess
import numpy as np, cv2

P = sys.argv[1]
ARG = {a.split("=")[0]: (a.split("=")[1] if "=" in a else True) for a in sys.argv[2:]}
# Blink frames: the approved pair lives in assets/ (blink_half.png, blink_closed.png). --half= / --closed= name a
# candidate in candidates/blink/ instead, for trying another pair.
HALF = ARG.get("--half", "assets:blink_half"); CLOSED = ARG.get("--closed", "assets:blink_closed")

# ================================================================== timetable and constants
FPS, N = 30, 300                      # 10 s loop; frame 300 == frame 0
S = 2048 / 954.0                      # reference pixels -> our pixels
TAU = 2 * np.pi

# --- light bands (step 3, unchanged)
# Reference: top 19/27, mid 17/26, low 23/15. Slowed by about 10% after review. Laps must stay whole numbers for the
# loop to close, and the two layers of a band must share no common factor (or the pattern repeats early).
LAPS = {"top": (17, 24), "mid": (15, 23), "low": (21, 13)}
MIX = {"top": (0.62, 0.38), "mid": (0.65, 0.35), "low": (0.75, 0.25)}
BAND_GAIN = {"top": 1.55, "mid": 0.80, "low": 0.97}
BAND_GAMMA = {"top": 1.0, "mid": 1.44, "low": 1.0}
GAMMA_REF, XFADE, MED_X, BLUR_X, SEAM_SIGMA, SEAM_BLUR = 200.0, 6, 41, 9.0, 60.0, 45.0

# --- street lights. Frame at which each one is opposite the centre window (measured: 0.7, 2.0, 3.3, 8.9 s)
PASS_FRAMES = [21, 60, 99, 267]
PASS_SIGMA = 12.3                     # frames; brightness of one pass = exp(-((f-f0)/sigma)^2)
PATCH_SPEED = 21.0 * S                # px/frame: the window-shaped patch of light slides right inside the bus
WIN_DELAY = {"L": 5.6, "C": 0.0, "R": -5.6}       # the light reaches the right window first, the left one last
PATCH_SOFT = 60.0                     # px, softness of the pillar shadow edge
KNEE = 228.0                          # above this, highlights roll off instead of clipping flat
# --- tunnel (measured 4.05 - 7.55 s): lights every 6 frames, no visible sweep
TUN_IN, TUN_OUT, TUN_IN_LEN, TUN_OUT_SIGMA = 120.0, 222.0, 9.0, 11.0
TUN_RIPPLE, TUN_PERIOD, TUN_PHASE = 0.08, 6.0, 126.0
# --- what the light does to a pixel inside the glass: out = base*(1+g*L)+a*L  (B,G,R), fitted on the reference
G_PASS, A_PASS = np.array([0.27, 0.44, 0.41], np.float32), np.array([13.0, 18.7, 28.9], np.float32)
G_TUN, A_TUN = np.array([0.26, 0.59, 0.43], np.float32), np.array([16.1, 27.1, 54.8], np.float32)
ADD_DARK, ADD_LUM = 0.47, (15.0, 40.0)   # the far, dark cabin catches less of the added light than cloth and skin
A_BODY_PASS, A_BODY_TUN = np.array([2.0, 5.0, 7.0], np.float32), np.array([4.0, 10.0, 16.0], np.float32)
REFL_PASS, REFL_TUN = 0.70, 0.85      # fixed reflections on the body get this much brighter (glints: 135 -> 200-230)
REFL_BLOOM, REFL_BLOOM_SIGMA = 0.55, 9.0   # and flare softly around themselves
# --- her skin in the still is a little brighter than the reference's skin between lamps (ours 109,154,203;
# reference 100,138,171). Bring the bright parts of her down to that, so the lamps have room to light her up.
GIRL_REST = np.array([0.92, 0.90, 0.85], np.float32)     # B,G,R factor for skin-bright pixels; hair is left alone
GIRL_REST_LUM = (90.0, 160.0)
BANDS_PASS, BANDS_TUN = 0.50, 0.75    # the bands themselves get brighter with the light

# --- road joints (measured starts: 1.17, 3.97, 6.33, 8.53 s). Each is a double hit; the picture rings at 4 frames/cycle
#           frame  amp1  phase1   gap  amp2   strength for curtains / girl      (amps in reference px)
JOINTS = [(35, 2.0, -np.pi / 4, 4, 2.8, 0.90), (119, 2.7, -np.pi / 4, 6, 1.8, 1.00),
          (190, 1.8, 0.0, 4, 1.2, 0.60), (256, 3.5, 0.0, 8, 1.2, 0.85)]
HIT_PERIOD, HIT_TAU = 4.0, 6.0
# --- steady road vibration of the whole picture (cycles per loop, amplitude in reference px, phase)
VIB_X = [(19, 0.33, 0.6)]
VIB_Y = [(31, 0.61, 0.0), (47, 0.38, 1.1)]
# --- curtains: a damped swing after each joint, bigger towards the hem.  name: (amp px at hem, period, decay, delay, idle)
CURTAINS = {"L": (13.0, 50.0, 32.0, 10.0, 7), "CL": (7.0, 44.0, 30.0, 9.0, 11), "CR": (12.5, 48.0, 32.0, 10.0, 8), "R": (10.5, 46.0, 30.0, 8.0, 13)}
CURTAIN_POW, CURTAIN_IDLE = 1.5, 0.6
# The swing must stop at the cloth's own edge where something fixed is drawn next to it (the far-side window frame).
# name: (x where the swing starts, x where it is full) on the left side, (x where it is still full, x where it ends) on the right
CURTAIN_XLIM = {"CL": (None, (640.0, 664.0)), "CR": (None, None), "R": (None, (1905.0, 1928.0)), "L": (None, None)}
# The right curtain of the centre window hangs in front of the far-side window frame; its free edge is the slanted
# line traced here as (y, x). The frame (lines at x = 1408 and 1437) must not move, so the swing is zero up to
# max(CR_X_MIN, edge - CR_BEFORE) and full from there + CR_RAMP.
CR_EDGE = [(300, 1432), (400, 1432), (450, 1440), (500, 1452), (560, 1463), (620, 1473), (680, 1483), (740, 1492),
           (800, 1500), (860, 1508), (920, 1514), (1000, 1520), (1060, 1528), (1160, 1535)]
CR_X_MIN, CR_BEFORE, CR_RAMP = 1444.0, 14.0, 20.0
# --- the girl: a quick lean then a short damped sway, bigger towards the head
GIRL_AMP, GIRL_PERIOD, GIRL_TAU, GIRL_POW, GIRL_IDLE = 6.0, 21.0, 14.0, 1.3, 0.4
GIRL_TOP, GIRL_BOTTOM = 520.0, 1150.0
# --- blink: half, closed x4, half. The reference blinks at frames 77, 239 and again at 251 (a double blink).
# After review: no double blink, one blink each time -> the 251 one is dropped.
BLINKS = [77, 239]
# --- breath on the glass: every 100 frames, fullest at frames 26 / 126 / 226
FOG_CENTER, FOG_RADII, FOG_ALPHA = (956.0, 880.0), (62.0, 36.0), 0.20
FOG_PERIOD, FOG_START, FOG_RISE, FOG_FALL = 100, 10, 16, 36
FOG_COLOR = np.array([176.0, 190.0, 186.0], np.float32)

# ================================================================== inputs
MD = os.path.join(P, "masks")
plate = np.load(os.path.join(MD, "plate_f32.npy"))
band = np.load(os.path.join(MD, "band_layer_f32.npy"))
info = json.load(open(os.path.join(MD, "masks.json")))
still = cv2.imread(os.path.join(P, "assets", "still.png"))
H, W = plate.shape[:2]
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def mask(name):
    return cv2.imread(os.path.join(MD, name), 0).astype(np.float32) / 255


def soft(m, sigma):
    return cv2.GaussianBlur(m, (0, 0), sigma)


glass, girl, refl = mask("glass.png"), mask("girl.png"), mask("body_refl.png")
glass_s = soft(glass, 2.0)
refl_s = soft(refl, 1.5)
body_s = 1 - soft(cv2.dilate(glass, np.ones((9, 9), np.uint8)), 3.0)

# eyes: the three states differ from the still only around the eyes
girl_s = soft(girl, 3.0)


def rest_dim(img):
    lum = img.mean(2)
    k = np.clip((lum - GIRL_REST_LUM[0]) / (GIRL_REST_LUM[1] - GIRL_REST_LUM[0]), 0, 1)
    k = (k * k * (3 - 2 * k) * girl_s)[..., None]
    return img * (1 - k * (1 - GIRL_REST))


refl_layer = plate * refl_s[..., None]
refl_bloom = cv2.GaussianBlur(refl_layer, (0, 0), REFL_BLOOM_SIGMA)
eye_frames = {}
for key, nm in (("half", HALF), ("closed", CLOSED)):
    im = cv2.imread(os.path.join(P, "assets", nm[7:] + ".png") if nm.startswith("assets:") else os.path.join(P, "candidates", "blink", nm + ".png"))
    ys, xs = np.where((im != still).any(2))
    box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    eye_frames[key] = (box, im[box[0]:box[1], box[2]:box[3]].astype(np.float32) - still[box[0]:box[1], box[2]:box[3]].astype(np.float32))
EYE_STATE = {}
for b in BLINKS:
    EYE_STATE[(b - 1) % N] = "half"
    for k in range(4):
        EYE_STATE[(b + k) % N] = "closed"
    EYE_STATE[(b + 4) % N] = "half"

# ================================================================== light bands layer (as in step 3)
R0, R1 = info["band_rows"]; split = info["band_split"]
ya, yb = R0 - 14, R1 + 15
rows = np.arange(ya, yb, dtype=np.float32)


def ramp(y, c):
    return np.clip((y - (c - XFADE / 2)) / XFADE, 0, 1)


def wrap_pad(a, n):
    return np.concatenate([a[:, -n:], a, a[:, :n]], axis=1)


def smooth_along_row(b):
    n = MED_X // 2
    med = np.median(np.lib.stride_tricks.sliding_window_view(wrap_pad(b, n), MED_X, axis=1), axis=-1).astype(np.float32)
    m = int(BLUR_X * 4)
    g = cv2.GaussianBlur(wrap_pad(med, m), (0, 0), sigmaX=BLUR_X, sigmaY=0.01)[:, m:-m]
    m2 = int(SEAM_BLUR * 3)
    far = cv2.GaussianBlur(wrap_pad(g, m2), (0, 0), sigmaX=SEAM_BLUR, sigmaY=0.01)[:, m2:-m2]
    x = np.arange(W, dtype=np.float32)
    w = np.exp(-np.minimum(x, W - 1 - x) ** 2 / (2 * SEAM_SIGMA ** 2))[None, :, None]
    return g * (1 - w) + far * w


c1, c2 = split["top"][1], split["mid"][1]
wrow = {"top": 1 - ramp(rows, c1), "mid": ramp(rows, c1) * (1 - ramp(rows, c2)), "low": ramp(rows, c2)}
band_rows = smooth_along_row(np.ascontiguousarray(band[ya:yb]))
layers = {}
for name in ("top", "mid", "low"):
    b = band_rows * wrow[name][:, None, None]
    amp = b.max(2, keepdims=True)
    b = b * np.power(np.clip(amp / GAMMA_REF, 1e-4, None), BAND_GAMMA[name] - 1.0) * BAND_GAIN[name]
    layers[name] = (np.ascontiguousarray(b), np.ascontiguousarray(b[:, ::-1]))


def band_frame(i):
    out = np.zeros((yb - ya, W, 3), np.float32)
    for name, (bo, bf) in layers.items():
        lo, lf = LAPS[name]
        wo, wf = MIX[name]
        out += wo * np.roll(bo, -(int(round(lo * W * i / N)) % W), axis=1) + wf * np.roll(bf, -(int(round(lf * W * i / N)) % W), axis=1)
    return out


# ================================================================== time functions (all periodic in N)
def circ(d):                           # signed circular distance in frames
    return (d + N / 2) % N - N / 2


def pass_intensity(f, delay=0.0):
    return [float(np.exp(-(circ(f - f0 - delay) / PASS_SIGMA) ** 2)) for f0 in PASS_FRAMES]


def tunnel_level(f):
    f = f % N
    if f < TUN_IN:
        lvl = 0.0
    elif f < TUN_IN + TUN_IN_LEN:
        u = (f - TUN_IN) / TUN_IN_LEN
        lvl = u * u * (3 - 2 * u)
    elif f <= TUN_OUT:
        lvl = 1.0
    else:
        lvl = float(np.exp(-((f - TUN_OUT) / TUN_OUT_SIGMA) ** 2))
    return lvl * (1 + TUN_RIPPLE * np.cos(TAU * (f - TUN_PHASE) / TUN_PERIOD))


def after(f, f0):                      # frames since an event, on the loop; negative = it has not happened "recently"
    u = (f - f0) % N
    return u if u < N * 0.8 else u - N


def shake(f):
    dx = sum(a * S * np.sin(TAU * c * f / N + ph) for c, a, ph in VIB_X)
    dy = sum(a * S * np.sin(TAU * c * f / N + ph) for c, a, ph in VIB_Y)
    for f0, a1, ph1, gap, a2, _ in JOINTS:
        for fa, a in ((f0, a1), (f0 + gap, a2)):
            u = after(f, fa)
            if u >= 0:
                dy += a * S * np.exp(-u / HIT_TAU) * np.cos(TAU * u / HIT_PERIOD + ph1)
    return float(dx), float(dy)


def swing(f, period, tau, delay):
    v = 0.0
    for f0, *_, strength in JOINTS:
        u = after(f, f0 + delay)
        if u >= 0:
            v += strength * np.exp(-u / tau) * np.sin(TAU * u / period)
    return v


# ================================================================== spatial weights for the sway
glass_in = soft(cv2.erode(glass, np.ones((15, 15), np.uint8)), 6.0)        # nothing may move at the window frame
W_CURT = {}
for name in CURTAINS:
    m = mask(f"curtain_{name}.png")
    ys = np.where(m.any(1))[0]
    prof = np.clip((YY - ys.min()) / float(ys.max() - ys.min()), 0, 1) ** CURTAIN_POW
    w_ = soft(cv2.dilate(m, np.ones((21, 21), np.uint8)), 10.0) * prof * glass_in * (1 - soft(girl, 8.0))
    lo, hi = CURTAIN_XLIM[name]
    if lo:
        u = np.clip((XX - lo[0]) / (lo[1] - lo[0]), 0, 1); w_ = w_ * (u * u * (3 - 2 * u))
    if hi:
        u = np.clip((hi[1] - XX) / (hi[1] - hi[0]), 0, 1); w_ = w_ * (u * u * (3 - 2 * u))
    if name == "CR":
        ey = np.interp(YY[:, 0], [e[0] for e in CR_EDGE], [e[1] for e in CR_EDGE]).astype(np.float32)
        CR_X0 = np.maximum(CR_X_MIN, ey - CR_BEFORE)                 # per row: swing is zero up to here
        u = np.clip((XX - CR_X0[:, None]) / CR_RAMP, 0, 1); w_ = w_ * (u * u * (3 - 2 * u))
    W_CURT[name] = w_
W_GIRL = soft(cv2.dilate(girl, np.ones((9, 9), np.uint8)), 12.0) * np.clip((GIRL_BOTTOM - YY) / (GIRL_BOTTOM - GIRL_TOP), 0, 1) ** GIRL_POW * glass_in

# window openings for the light patch (x extents; the left and right windows continue outside the picture)
wC = info["windows"]["C"]; width = wC["right"] - wC["left"]
WIN_X = {"C": (wC["left"], wC["right"]), "L": (info["windows"]["L"]["right"] - width, info["windows"]["L"]["right"]),
         "R": (info["windows"]["R"]["left"], info["windows"]["R"]["left"] + width)}
WIN_COLS = {"L": XX < 470, "C": (XX >= 470) & (XX < 1730), "R": XX >= 1730}

# fog shape
fog_d = np.sqrt(((XX - FOG_CENTER[0]) / FOG_RADII[0]) ** 2 + ((YY - FOG_CENTER[1]) / FOG_RADII[1]) ** 2)


def fog_state(f):
    u = (f - FOG_START) % FOG_PERIOD
    if u < FOG_RISE:
        p = np.sin(0.5 * np.pi * u / FOG_RISE) ** 2
        return float(p), 0.65 + 0.35 * float(p)
    if u < FOG_RISE + FOG_FALL:
        v = (u - FOG_RISE) / FOG_FALL
        return float(1 - v * v * (3 - 2 * v)), 1.0
    return 0.0, 1.0


def light_field(f):
    """Lp: street-light patch (per pixel), Lt: tunnel level (scalar)."""
    Lp = np.zeros((H, W), np.float32)
    for wname, (a, b) in WIN_X.items():
        for f0, inten in zip(PASS_FRAMES, pass_intensity(f, WIN_DELAY[wname])):
            if inten < 0.01:
                continue
            s = PATCH_SPEED * circ(f - f0 - WIN_DELAY[wname])
            lit = np.clip((XX - (a + s)) / PATCH_SOFT + 0.5, 0, 1) * np.clip(((b + s) - XX) / PATCH_SOFT + 0.5, 0, 1)
            Lp += inten * lit * WIN_COLS[wname]
    return Lp, tunnel_level(f)


def disp_field(f):
    D = np.zeros((H, W), np.float32)
    for name, (amp, period, tau, delay, idle) in CURTAINS.items():
        D += W_CURT[name] * (-amp * swing(f, period, tau, delay) + CURTAIN_IDLE * np.sin(TAU * idle * f / N + idle))
    D += W_GIRL * (GIRL_AMP * swing(f, GIRL_PERIOD, GIRL_TAU, 0.0) + GIRL_IDLE * np.sin(TAU * 9 * f / N))
    return D


def frame(i, parts=False):
    f = float(i % N)
    base = plate.copy()
    st = EYE_STATE.get(int(i % N))
    if st:
        (y0, y1, x0, x1), d = eye_frames[st]
        base[y0:y1, x0:x1] += d
    base = rest_dim(base)
    # --- sway: one horizontal displacement field for curtains and girl
    D = disp_field(f)
    img = cv2.remap(base, XX - D, YY, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # --- light inside the windows
    Lp, Lt = light_field(f)
    gL = (Lp[..., None] * G_PASS + Lt * G_TUN) * glass_s[..., None]
    kd = np.clip((img.mean(2) - ADD_LUM[0]) / (ADD_LUM[1] - ADD_LUM[0]), 0, 1)
    kd = ADD_DARK + (1 - ADD_DARK) * kd * kd * (3 - 2 * kd)
    aL = (Lp[..., None] * A_PASS + Lt * A_TUN) * (glass_s * kd)[..., None]
    img = img * (1 + gL) + aL
    # --- breath on the glass
    fa, fr_ = fog_state(f)
    if fa > 0:
        Lloc = np.clip(Lp + Lt, 0, 1.5)
        a = (FOG_ALPHA * fa * np.exp(-(fog_d / fr_) ** 2 * 1.4) * glass_s)[..., None]
        img = img * (1 - a) + FOG_COLOR * (0.8 + 0.3 * Lloc[..., None]) * a
    # --- body and its fixed reflections
    Ig = sum(pass_intensity(f))
    img += (A_BODY_PASS * Ig + A_BODY_TUN * Lt) * body_s[..., None]
    rg = REFL_PASS * Ig + REFL_TUN * Lt
    img += refl_layer * rg + refl_bloom * (REFL_BLOOM * rg)
    # --- flowing bands, in front of everything
    img[ya:yb] += band_frame(i) * (1 + BANDS_PASS * Ig + BANDS_TUN * Lt)
    # --- the whole picture shakes
    dx, dy = shake(f)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    hi = out > KNEE
    out[hi] = KNEE + (255.0 - KNEE) * np.tanh((out[hi] - KNEE) / 60.0)
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    if parts:
        return out, dict(dx=dx, dy=dy, Ig=Ig, Lt=Lt, fog=fa, eye=st or "open", Dmin=float(D.min()), Dmax=float(D.max()))
    return out


# ================================================================== checks before encoding
os.makedirs(os.path.join(P, "out"), exist_ok=True)
f0, f300 = frame(0), frame(N)
f1, f299 = frame(1), frame(N - 1)
curve = [frame(i, parts=True)[1] for i in range(0, N)] if "--curves" in ARG else None
checks = dict(frames=N, fps=FPS, size=[W, H], laps=LAPS,
              curtain_sway_weight_inside_right_curtain_max=round(float(W_CURT["CR"].max()), 3), blink_frames={k: v for k, v in sorted(EYE_STATE.items())}, eyes=dict(half=HALF, closed=CLOSED),
              loop_frame300_equals_frame0=bool(np.array_equal(f0, f300)),
              mean_abs_diff_f0_f1=round(float(np.abs(f1.astype(np.int16) - f0).mean()), 3),
              mean_abs_diff_seam_f299_f0=round(float(np.abs(f0.astype(np.int16) - f299).mean()), 3),
              shake_px={"max_abs_dx": round(max(abs(shake(i)[0]) for i in range(N)), 2), "max_abs_dy": round(max(abs(shake(i)[1]) for i in range(N)), 2)})
# the far-side window frame must not move with curtains or girl: largest horizontal shift over the whole loop
FAR_FRAME = ((XX >= 1100) & (XX <= CR_X0[:, None])).astype(np.float32)       # cabin + far frame, up to the curtain's edge zone
mx = dict(far_frame_right_part=0.0, far_frame_top_bar_left_part=0.0, right_curtain=0.0, left_curtain=0.0, girl_head=0.0)
for i in range(N):
    Dm = np.abs(disp_field(float(i)))
    mx["far_frame_right_part"] = max(mx["far_frame_right_part"], float((Dm * FAR_FRAME)[300:1160].max()))
    mx["far_frame_top_bar_left_part"] = max(mx["far_frame_top_bar_left_part"], float(Dm[300:470, 692:1100].max()))
    mx["right_curtain"] = max(mx["right_curtain"], float(Dm[300:1160, 1470:1620].max()))
    mx["left_curtain"] = max(mx["left_curtain"], float(Dm[300:1160, 100:350].max()))
    mx["girl_head"] = max(mx["girl_head"], float(Dm[520:800, 640:960].max()))
checks["max_sway_shift_px_over_loop"] = {k: round(v, 3) for k, v in mx.items()}
json.dump(checks, open(os.path.join(P, "out", "step4_checks.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in checks.items() if k != "blink_frames"}, ensure_ascii=False))
if "--dump" in ARG:
    for i in ARG["--dump"].split(","):
        im, info_ = frame(int(i), parts=True)
        cv2.imwrite(os.path.join(P, "out", f"step4_frame_{int(i):03d}.png"), im)
        print(i, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in info_.items()})
if "--check-only" in ARG:
    sys.exit(0)

OUT = ARG.get("--out", "final_picture")
master = os.path.join(P, "out", OUT + "_master.mp4")
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                       "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", master], stdin=subprocess.PIPE)
for i in range(N):
    ff.stdin.write(frame(i).tobytes())
ff.stdin.close(); ff.wait()
small = os.path.join(P, "out", OUT + "_1440.mp4")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", master, "-vf", "scale=1440:1080:flags=lanczos", "-c:v", "libx264",
                "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", small], check=True)
print("wrote", master, small)
