"""Measure how the light bands move in the reference video.
Usage: python -I measure_ref_bands.py <frames_dir>

A layer that wraps around the picture L times in the 300-frame loop makes the n-th spatial Fourier coefficient of a
row rotate n*L times in the loop, so the time-FFT of that coefficient has a sharp peak at bin n*L.
Positive bin = the layer moves to the LEFT. Static picture and lighting changes sit near bin 0."""
import sys, glob, os
import numpy as np, cv2
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "f*.png")))
Y0, Y1 = 366, 498
stack = np.stack([cv2.imread(p)[Y0:Y1].astype(np.float32).mean(2) for p in files])     # T,h,W
T, h, W = stack.shape
print("frames", T, "width", W)
for (a, b) in ((380, 392), (392, 404), (404, 416), (416, 428), (428, 440), (440, 452), (452, 464), (464, 476), (476, 490)):
    s = stack[:, a - Y0:b - Y0].mean(1)                     # T, W
    line = []
    for n in (1, 2, 3):
        X = (s * np.exp(-2j * np.pi * n * np.arange(W) / W)[None]).sum(1)      # T
        S = np.abs(np.fft.fft(X)) / T
        bins = np.fft.fftfreq(T, 1 / T).astype(int)
        S[np.abs(bins) <= 3] = 0                              # drop static + slow lighting
        top = np.argsort(-S)[:5]
        line.append(f"n={n}: " + " ".join(f"{bins[i]/n:+6.1f}({S[i]:.0f})" for i in top))
    print(f"rows {a}-{b}  laps -> " + " | ".join(line))
