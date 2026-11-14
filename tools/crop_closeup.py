"""Close-up of the original girl (drawing-style reference). Usage: python -I crop_closeup.py <frame.png> <out.png>"""
import sys, cv2
im = cv2.imread(sys.argv[1])
c = im[222:446, 258:486]
cv2.imwrite(sys.argv[2], cv2.resize(c, None, fx=5, fy=5, interpolation=cv2.INTER_LANCZOS4))
print(c.shape)
