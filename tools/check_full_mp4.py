"""Measure the exported step-4 video with the same tools as the reference and put the numbers side by side.
Usage: python check_full_mp4.py <project_dir>   (expects work/step4_frames/f000.png ...)"""
import sys, glob, os, json
import numpy as np, cv2
P = sys.argv[1]
files = sorted(glob.glob(P + "/work/step4_frames/f*.png"))
fr = np.stack([cv2.imread(p) for p in files]).astype(np.float32)
T, H, W, _ = fr.shape
k = W / 954.0
lum = fr.mean(3)
ref = np.load(P + "/work/ref_motion.npz")

def lap(a):
    return cv2.Laplacian(cv2.GaussianBlur(a, (0, 0), 1.0 * k), cv2.CV_32F, ksize=3)

def track(box):
    x0, y0, x1, y1 = [int(round(v * k)) for v in box]
    stack = np.stack([lap(lum[t, y0:y1, x0:x1]) for t in range(T)])
    r = np.median(stack, 0)
    win = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
    return np.array([cv2.phaseCorrelate(r, stack[t], win)[0] for t in range(T)], np.float32) / k     # in reference px

g = (track((0, 540, 954, 610)) + track((0, 96, 954, 134))) / 2
rg = ref["g"]
out = {}
out["shake_dy_corr_with_reference"] = round(float(np.corrcoef(g[:, 1], rg[:, 1])[0, 1]), 3)
out["shake_dy_rms_ours_vs_ref_px954"] = [round(float(g[:, 1].std()), 2), round(float(rg[:, 1].std()), 2)]
out["shake_dy_peak_ours_vs_ref_px954"] = [round(float(np.abs(g[:, 1]).max()), 2), round(float(np.abs(rg[:, 1]).max()), 2)]
out["shake_dx_rms_ours_vs_ref_px954"] = [round(float(g[:, 0].std()), 2), round(float(rg[:, 0].std()), 2)]
out["joint_peak_frames_ours"] = [int(a + np.argmax(np.abs(g[a:a + 20, 1]))) for a in (30, 114, 184, 252)]
out["joint_peak_frames_ref"] = [int(a + np.argmax(np.abs(rg[a:a + 20, 1]))) for a in (30, 114, 184, 252)]
for nm, box in {"curtCR_low": (655, 468, 760, 532), "girl_head": (285, 235, 455, 378)}.items():
    rel = track(box)[:, 0] - g[:, 0]
    rrel = ref["tr_" + nm][:, 0] - rg[:, 0]
    out[f"sway_{nm}_range_ours_vs_ref_px954"] = [[round(float(rel.min()), 2), round(float(rel.max()), 2)], [round(float(rrel.min()), 2), round(float(rrel.max()), 2)]]
    out[f"sway_{nm}_corr"] = round(float(np.corrcoef(rel, rrel)[0, 1]), 3)
reg = {"cabin_dark": (480, 250, 640, 360), "curtCR": (670, 180, 750, 360), "curtL": (60, 180, 150, 360), "body": (150, 620, 800, 700),
       "skin": (385, 370, 399, 380)}
rest = list(range(40, 50)) + list(range(246, 256)); tun = list(range(150, 200))
light = {}
for nm, (x0, y0, x1, y1) in reg.items():
    v = fr[:, int(y0 * k):int(y1 * k), int(x0 * k):int(x1 * k)].reshape(T, -1, 3).mean(1)
    if nm == "skin":          # our cheek is not at the same place as the reference girl's: use our own patch
        v = fr[:, int(778 * W / 2048):int(800 * W / 2048), int(832 * W / 2048):int(872 * W / 2048)].reshape(T, -1, 3).mean(1)
        rv = None
    else:
        rv = ref["cur_" + nm]
    light[nm] = dict(ours_rest=v[rest].mean(0).round(0).tolist(), ours_pass=v[[20, 21, 60, 61]].mean(0).round(0).tolist(), ours_tunnel=v[tun].mean(0).round(0).tolist())
    if rv is not None:
        light[nm].update(ref_rest=rv[rest].mean(0).round(0).tolist(), ref_pass=rv[[20, 21, 60, 61]].mean(0).round(0).tolist(),
                         ref_tunnel=rv[tun].mean(0).round(0).tolist(), corr_lum=round(float(np.corrcoef(v.mean(1), rv.mean(1))[0, 1]), 3))
out["lighting_BGR"] = light
# blink
e = lum[:, int(700 * W / 2048):int(775 * W / 2048), int(770 * W / 2048):int(835 * W / 2048)].mean(axis=(1, 2))
c = lum[:, int(778 * W / 2048):int(800 * W / 2048), int(832 * W / 2048):int(872 * W / 2048)].mean(axis=(1, 2))
r = e / c
thr = np.median(r) + 0.6 * (r.max() - np.median(r))
out["blink_closed_frames_ours"] = np.where(r > thr)[0].tolist()
out["blink_closed_frames_ref"] = [77, 78, 79, 80, 239, 240, 241, 242, 251, 252, 253, 254]
d = lambda a, b: round(float(np.abs(fr[a] - fr[b]).mean()), 2)
out["mean_abs_diff"] = dict(f0_f1=d(0, 1), f149_f150=d(149, 150), f298_f299=d(298, 299), seam_f299_f0=d(299, 0))
c_ = json.load(open(P + "/out/step4_checks.json")); c_["measured_on_mp4"] = out
json.dump(c_, open(P + "/out/step4_checks.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
