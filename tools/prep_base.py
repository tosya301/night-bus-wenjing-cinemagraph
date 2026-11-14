"""Make Image 1 for gpt-image: one reference frame, author's logo painted out, upscaled x2.
Usage: python -I prep_base.py <frame.png> <out.png> <check_crop.png>"""
import sys
import numpy as np, cv2
src, out, chk = sys.argv[1:4]
im = cv2.imread(src)
h, w = im.shape[:2]
x0, y0, x1, y1 = 712, 636, 942, 708          # logo + two text lines, bottom right
mask = np.zeros((h, w), np.uint8); mask[y0:y1, x0:x1] = 255
clean = cv2.inpaint(im, mask, 6, cv2.INPAINT_TELEA)
# keep the film-like grain of the neighbourhood instead of a flat inpaint patch
rng = np.random.default_rng(7)
ref = im[y0:y1, x0-240:x1-240].astype(np.float32)
grain = ref - cv2.GaussianBlur(ref, (0, 0), 3)
patch = clean[y0:y1, x0:x1].astype(np.float32) + grain
clean[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
assert np.array_equal(clean[mask == 0], im[mask == 0])
big = cv2.resize(clean, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(out, big)
cv2.imwrite(chk, np.hstack([im[560:720, 560:954], clean[560:720, 560:954]]))
print("saved", out, big.shape)
