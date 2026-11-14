"""Measure the reference video for step 4: shake, sway, lighting, blink frames, glass fog, reflections.
Usage: python -I measure_ref_motion.py <project_dir>
Writes work/ref_motion.npz (per-frame curves), work/ref_kymo_*.png (x-t images), work/ref_fog_map.png"""
import sys, glob, os
import numpy as np, cv2

P = sys.argv[1]
files = sorted(glob.glob(os.path.join(P, "refs", "frames", "f*.png")))
fr = np.stack([cv2.imread(p) for p in files]).astype(np.float32)          # T,H,W,3
T, H, W, _ = fr.shape
lum = fr.mean(3)


def lap(a):                      # line work only: insensitive to lighting level
    return cv2.Laplacian(cv2.GaussianBlur(a, (0, 0), 1.0), cv2.CV_32F, ksize=3)


def track(box, ref=None):
    x0, y0, x1, y1 = box
    stack = np.stack([lap(lum[t, y0:y1, x0:x1]) for t in range(T)])
    if ref is None:
        ref = np.median(stack, 0)
    win = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
    out = np.zeros((T, 3), np.float32)
    for t in range(T):
        (dx, dy), resp = cv2.phaseCorrelate(ref, stack[t], win)
        out[t] = (dx, dy, resp)
    return out


boxes = {
    "body_bottom": (0, 540, 954, 610), "body_top": (0, 96, 954, 134), "body_low": (150, 600, 800, 700),
    "curtL_low": (40, 468, 175, 532), "curtL_up": (40, 160, 165, 270), "curtL_mid": (60, 280, 170, 370),
    "curtCR_low": (655, 468, 760, 532), "curtCR_up": (655, 160, 760, 270),
    "curtR_up": (850, 160, 905, 300),
    "girl_head": (285, 235, 455, 378), "girl_body": (275, 468, 400, 530),
}
tr = {k: track(b) for k, b in boxes.items()}
g = (tr["body_bottom"][:, :2] + tr["body_top"][:, :2]) / 2
print("global shake (body top+bottom), px at 954 wide:  dx min/max %.2f %.2f  dy min/max %.2f %.2f  rms dx %.2f dy %.2f" % (
    g[:, 0].min(), g[:, 0].max(), g[:, 1].min(), g[:, 1].max(), g[:, 0].std(), g[:, 1].std()))
print("agreement top vs bottom: rms diff dx %.2f dy %.2f" % ((tr["body_bottom"][:, 0] - tr["body_top"][:, 0]).std(),
                                                             (tr["body_bottom"][:, 1] - tr["body_top"][:, 1]).std()))
for k in boxes:
    if k.startswith("body"):
        continue
    rel = tr[k][:, :2] - g
    print(f"{k:11s} relative to body: dx rms {rel[:, 0].std():.2f} range {rel[:, 0].min():+.2f}..{rel[:, 0].max():+.2f}   dy rms {rel[:, 1].std():.2f}   resp {tr[k][:, 2].mean():.2f}")
print("global dy every 2 frames x100:")
print(" ".join(f"{int(round(v * 100)):d}" for v in g[::2, 1]))
print("global dx every 2 frames x100:")
print(" ".join(f"{int(round(v * 100)):d}" for v in g[::2, 0]))

# ---- lighting curves
reg = {"cabin_dark": (480, 250, 640, 360), "curtCR": (670, 180, 750, 360), "curtL": (60, 180, 150, 360),
       "face": (355, 318, 430, 372), "body": (150, 620, 800, 700), "stripe": (150, 580, 700, 590),
       "top_trim_hl": (190, 112, 440, 118), "glintL": (188, 395, 205, 440), "glintR": (800, 395, 850, 440),
       "bands_cabin": (480, 380, 650, 490)}
cur = {}
for k, (x0, y0, x1, y1) in reg.items():
    cur[k] = fr[:, y0:y1, x0:x1].reshape(T, -1, 3).mean(1)           # T,3 (B,G,R)
for k in ("cabin_dark", "curtCR", "curtL", "face", "body", "stripe", "glintL", "glintR", "bands_cabin"):
    v = cur[k].mean(1)
    print(f"{k:11s} lum every 5 frames: " + " ".join(f"{int(x)}" for x in v[::5]))
print("curtCR R/B every 5 frames x100: " + " ".join(f"{int(100 * a / max(b, 1))}" for a, b in zip(cur['curtCR'][::5, 2], cur['curtCR'][::5, 0])))

# ---- blink: iris box gets skin-coloured when the eye closes
eye = lum[:, 338:356, 362:384].mean(1).mean(1); cheek = lum[:, 372:386, 372:420].mean(1).mean(1)
ratio = eye / np.maximum(cheek, 1)
base = np.median(ratio)
bl = np.where(ratio > base + 0.5 * (ratio.max() - base))[0]
print("blink: eye/cheek ratio median %.3f max %.3f; frames above half-way:" % (base, ratio.max()), bl.tolist())
print("ratio x100 around them:", {int(f): [int(100 * ratio[i]) for i in range(max(0, f - 3), min(T, f + 5))] for f in (bl[:1].tolist() + [b for a, b in zip(bl[:-1], bl[1:]) if b - a > 3])})

# ---- kymographs (x-t): upper glass rows, face rows, lower glass rows
def kymo(y0, y1, name):
    k = lum[:, y0:y1].mean(1)                      # T,W
    img = np.clip(k * 2.2, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(P, "work", f"ref_kymo_{name}.png"), cv2.resize(img, (W, T * 2), interpolation=cv2.INTER_NEAREST))
kymo(190, 360, "upper_glass"); kymo(320, 372, "face_rows"); kymo(490, 530, "lower_glass")

# ---- glass fog: 3 cycles per loop. Amplitude and phase of temporal bin 3 for every pixel
small = np.stack([cv2.GaussianBlur(lum[t], (0, 0), 2.0) for t in range(T)], axis=2)       # H,W,T
F3 = (small * np.exp(-2j * np.pi * 3 * np.arange(T) / T)[None, None]).sum(2) / T
amp3 = np.abs(F3)
cv2.imwrite(os.path.join(P, "work", "ref_fog_map.png"), np.clip(amp3 * 40, 0, 255).astype(np.uint8))
ys, xs = np.unravel_index(np.argsort(-amp3[150:540, 280:780].ravel())[:400], (390, 500))
print("strongest 3-per-loop pixels in the centre window: x %d-%d (median %d)  y %d-%d (median %d)  amp max %.1f" % (
    xs.min() + 280, xs.max() + 280, np.median(xs) + 280, ys.min() + 150, ys.max() + 150, np.median(ys) + 150, amp3[150:540, 280:780].max()))
np.save(os.path.join(P, "work", "ref_fog_F3.npy"), F3.astype(np.complex64))
for nm, (a, b) in {"joint 1.2s": (30, 52), "joint 4.0s": (114, 138), "joint 6.4s": (184, 208), "joint 8.6s": (252, 276)}.items():
    print(nm, "dy x100 frames %d-%d:" % (a, b - 1), " ".join(f"{int(round(v * 100))}" for v in g[a:b, 1]))
for k in ("curtCR_low", "curtL_mid", "girl_head"):
    rel = tr[k][:, 0] - g[:, 0]
    print(k, "dx rel x100 every 3 frames:", " ".join(f"{int(round(v * 100))}" for v in rel[::3]))
np.savez(os.path.join(P, "work", "ref_motion.npz"), g=g, ratio=ratio, **{f"tr_{k}": v for k, v in tr.items()}, **{f"cur_{k}": v for k, v in cur.items()})
