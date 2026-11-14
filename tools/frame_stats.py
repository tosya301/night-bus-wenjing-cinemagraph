"""Per-frame brightness of the reference video: whole frame, centre-window glass, girl's face.
Usage: python -I frame_stats.py <frames_dir> <out_csv>"""
import sys, glob, os
import numpy as np, cv2
d, out = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(d, "f*.png"))):
    im = cv2.imread(p).astype(np.float32)
    y = im.mean(axis=2)
    h, w = y.shape
    rows.append((os.path.basename(p), y.mean(), y[int(.20*h):int(.72*h), int(.29*w):int(.78*w)].mean(),
                 y[int(.40*h):int(.56*h), int(.33*w):int(.46*w)].mean()))
with open(out, "w") as f:
    f.write("frame,all,glass,face\n")
    for r in rows:
        f.write("%s,%.2f,%.2f,%.2f\n" % r)
a = np.array([r[1:] for r in rows])
print("frames", len(rows), "mean all/glass/face", a.mean(0).round(1))
for i in range(0, len(rows), 10):
    print(i, a[i].round(1))
