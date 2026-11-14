"""Step 2: split the still into parts (masks) and peel the light bands.

Usage: python build_masks.py <project_dir>

Reads  assets/still.png
Writes masks/glass.png, curtain_{L,CL,CR,R}.png, curtains.png, girl.png, bands.png,
       band_{top,mid,low}.png, body_refl.png           (8-bit, 255 = inside)
       masks/plate.png        the still with the light bands removed ("under the bands")
       masks/band_layer.png   the bands alone on black (additive light)
       masks/masks.json       numbers used / measured
       masks/schematic.jpg    coloured overview for review
       masks/peel_check.jpg   original / plate / bands-only strips
"""
import sys, os, json
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

P = sys.argv[1]
MD = os.path.join(P, "masks")
os.makedirs(MD, exist_ok=True)
im = cv2.imread(os.path.join(P, "assets", "still.png"))
H, W = im.shape[:2]
f = im.astype(np.float32)
blur = cv2.GaussianBlur(f, (0, 0), 1.5)
b, g, r = blur[..., 0], blur[..., 1], blur[..., 2]
gb, rb, lum = g - b, r - b, blur.mean(2)

# ---------------------------------------------------------------- shapes read off the gridded crops
BAND_ROWS = (806, 1040)            # rows touched by the light bands (measured on the pillars)
WINDOWS = {                        # search boxes for each window opening
    "L": dict(x0=0, x1=376, y0=288, y1=1168, cut="left"),
    "C": dict(x0=560, x1=1642, y0=296, y1=1168, cut=None),
    "R": dict(x0=1812, x1=W, y0=302, y1=1168, cut="right"),
}
GIRL_POLY = [(578, 1154), (578, 1004), (594, 985), (590, 900), (592, 800), (600, 730), (622, 660), (655, 600),
             (700, 552), (760, 522), (830, 512), (890, 528), (940, 570), (975, 640), (995, 720), (998, 800),
             (990, 870), (962, 925), (925, 950), (905, 965), (915, 985), (950, 1010), (990, 1050), (1006, 1100),
             (1008, 1154)]
CURTAIN_POLYS = {
    "L":  [(88, 296), (366, 296), (368, 1158), (294, 1158), (286, 1010), (242, 905), (186, 705), (126, 505)],
    "CL": [(570, 304), (690, 304), (690, 1006), (570, 1006)],
    "CR": [(1398, 304), (1632, 304), (1632, 1158), (1408, 1158), (1412, 1090), (1400, 700)],
    "R":  [(1820, 310), (1942, 310), (1942, 1000), (1948, 1158), (1838, 1158), (1820, 1000)],
}


def poly_mask(pts):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    return m


def ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def medfilt1(a, k):
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    return np.array([np.median(ap[i:i + k]) for i in range(len(a))])


# ---------------------------------------------------------------- 1. window glass
# "light" = lit cloth / trim inside a window: brighter than the dark frame ring, not navy, not orange.
light = ((lum > 34) & (gb > -8) & (rb < 14)).astype(np.uint8) * 255
light = cv2.morphologyEx(light, cv2.MORPH_OPEN, ellipse(7))
glass = np.zeros((H, W), np.uint8)
win_info = {}
for name, w in WINDOWS.items():
    ys = np.arange(w["y0"], w["y1"])
    xl = np.full(len(ys), np.nan); xr = np.full(len(ys), np.nan)
    for i, y in enumerate(ys):
        if BAND_ROWS[0] <= y <= BAND_ROWS[1]:
            continue
        xs = np.where(light[y, w["x0"]:w["x1"]] > 0)[0]
        if len(xs) >= 20 and xs.max() - xs.min() > 60:
            xl[i], xr[i] = xs.min() + w["x0"], xs.max() + w["x0"]
    ok = ~np.isnan(xl)
    top, bot = ys[ok].min(), ys[ok].max()
    sel = (ys >= top) & (ys <= bot)
    ys2 = ys[sel]
    xl2 = np.interp(ys2, ys[ok], xl[ok]); xr2 = np.interp(ys2, ys[ok], xr[ok])
    # straight part: long median removes local dents (hair, braid, highlights); corners keep their shape
    core = (ys2 > top + 75) & (ys2 < bot - 75)
    xl_s, xr_s = medfilt1(xl2, 5), medfilt1(xr2, 5)
    xl_l, xr_l = medfilt1(xl2, 81), medfilt1(xr2, 81)
    xl_f = np.where(core, np.minimum(xl_s, xl_l + 2), xl_s)
    xr_f = np.where(core, np.maximum(xr_s, xr_l - 2), xr_s)
    xl_f = np.where(core, xl_l, xl_f); xr_f = np.where(core, xr_l, xr_f)
    if w["cut"] == "left":
        xl_f[:] = 0
    if w["cut"] == "right":
        xr_f[:] = W - 1
        upper = (ys2 > top + 75) & (ys2 < 700)
        x_side = float(np.median(xl2[upper]))
        rad = 52.0                                     # bottom-left corner radius, read off the gridded crop
        for i, y in enumerate(ys2):
            if y >= 700:
                dy = y - (bot - rad)
                xl_f[i] = x_side + (rad - np.sqrt(max(rad * rad - dy * dy, 0.0)) if dy > 0 else 0.0)
    for y, a, c in zip(ys2, xl_f, xr_f):
        glass[int(y), int(round(a)):int(round(c)) + 1] = 255
    win_info[name] = dict(top=int(top), bottom=int(bot), left=int(np.median(xl_f)), right=int(np.median(xr_f)))
glass = cv2.morphologyEx(glass, cv2.MORPH_CLOSE, ellipse(9))
glass = cv2.erode(glass, ellipse(5))          # stay off the dark frame ring

# ---------------------------------------------------------------- 2. girl
girl = cv2.bitwise_and(poly_mask(GIRL_POLY), glass)

# ---------------------------------------------------------------- 3. curtains
lightc = ((lum > 30) & (gb > -8)).astype(np.uint8) * 255
lightc[BAND_ROWS[0]:BAND_ROWS[1] + 1] = 255          # bands hide the cloth there: trust the polygon
lightc = cv2.morphologyEx(lightc, cv2.MORPH_CLOSE, ellipse(13))
curtains = {}
for name, pts in CURTAIN_POLYS.items():
    m = cv2.bitwise_and(poly_mask(pts), glass)
    if name == "L":
        m = cv2.bitwise_and(m, lightc)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ellipse(9))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = np.where(lab == keep, 255, 0).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        m = np.zeros_like(m); cv2.drawContours(m, cs, -1, 255, -1)        # no holes
    m[girl > 0] = 0
    curtains[name] = m
curt_all = np.zeros((H, W), np.uint8)
for m in curtains.values():
    curt_all |= m

# ---------------------------------------------------------------- 4. light bands: peel
# A band is a bright streak that is thin vertically and long horizontally. Three regions, three methods:
#   window interiors : fill the band rows from the clean rows above and below, keep whatever is darker
#   the girl         : short kernels only, so flat areas (cheek, neck, blouse) are never taken for a streak
#   pillars and body : keep the fixed glints, lift only what is long horizontally
KV_GIRL, KV_BIG, KH, KH_LONG = 61, 261, 91, 181
R0, R1 = BAND_ROWS
GIRL_BAND_ROWS = (806, 1004)       # on her the lowest band ends above the collar


def tophat_v(k):
    return np.clip(f - cv2.morphologyEx(f, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))), 0, None)


def open_h(a, k):
    return cv2.morphologyEx(a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))


def row_window(r0, r1):
    w_ = np.zeros(H, np.float32); w_[r0:r1 + 1] = 1
    return cv2.GaussianBlur(w_.reshape(-1, 1), (0, 0), 3).ravel()[:, None]


w_g = cv2.GaussianBlur(cv2.dilate(girl, ellipse(31)).astype(np.float32) / 255, (0, 0), 10)        # girl
w_i = cv2.GaussianBlur(cv2.erode(glass, ellipse(9)).astype(np.float32) / 255, (0, 0), 4) * (1 - w_g)  # interiors
w_p = np.clip(1 - w_g - w_i, 0, 1)                                                                 # pillars, body

# --- A. window interiors
top_a = f[R0 - 11:R0 - 1].mean(0); bot_a = f[R1 + 2:R1 + 12].mean(0)
tt = ((np.arange(H, dtype=np.float32) - (R0 - 1)) / float(R1 + 2 - (R0 - 1))).clip(0, 1)[:, None, None]
V = top_a[None] * (1 - tt) + bot_a[None] * tt
inrows = np.zeros((H, 1, 1), np.float32); inrows[R0:R1 + 1] = 1
band_A = np.clip(f - np.minimum(f, V), 0, None) * inrows

# --- B. pillars and body: one scalar ratio for all three channels (per-channel opening leaves rainbow residue)
diff_big = tophat_v(KV_BIG)
d = diff_big.max(2)
ratio_B = cv2.GaussianBlur(np.clip(open_h(d, KH) / (d + 1e-3), 0, 1), (0, 0), 1.5)
band_B = diff_big * (ratio_B * row_window(R0, R1))[..., None]

# --- C. the girl
diff_g = tophat_v(KV_GIRL)
dg = diff_g.max(2)
warmth = np.clip((diff_g[..., 2] - diff_g[..., 0]) / (dg + 1e-3), 0, 1)       # band light is orange,
warm_gate = np.clip((warmth - 0.15) / 0.25, 0, 1)                           # collar and cable are neutral
keep = np.maximum(open_h(dg * warm_gate, KH), open_h(dg, KH_LONG))           # orange and long, or very long
ratio_C = cv2.GaussianBlur(np.clip(keep / (dg + 1e-3), 0, 1), (0, 0), 1.5)
band_C = diff_g * (ratio_C * row_window(*GIRL_BAND_ROWS))[..., None]
# band light is smooth along the row; the column-by-column estimate is not (hair strands), and subtracting a
# hatched estimate would print the hatch into her neck. Smooth the estimate along x before using it.
band_C = cv2.blur(cv2.blur(band_C, (31, 1)), (31, 1))
# the hair-thin top streak (about 7 px): find its rows, rebuild what is under it from the rows just above and below
thin = tophat_v(21).max(2)
thin_m = (open_h(thin, KH) > 6).astype(np.uint8)
thin_m[:R0] = 0; thin_m[R0 + 40:] = 0                      # only the top streak
thin_m = cv2.dilate(thin_m, np.ones((7, 1), np.uint8))
plate_thin = f.copy()
for x in np.where(thin_m.any(0))[0]:
    ys = np.where(thin_m[:, x])[0]
    y0, y1 = ys.min() - 1, ys.max() + 1
    tcol = ((np.arange(y0, y1 + 1) - y0) / float(y1 - y0))[:, None]
    plate_thin[y0:y1 + 1, x] = f[y0, x][None] * (1 - tcol) + f[y1, x][None] * tcol
band_thin = np.clip(f - plate_thin, 0, None)
band_C = np.maximum(band_C, band_thin)

band = band_A * w_i[..., None] + band_B * w_p[..., None] + band_C * w_g[..., None]
plate = np.clip(f - band, 0, 255)

# --- colour of the band light where it sat on her: the red channel was already clipped on skin, so what was
# lifted there is pale green. For the layer that will flow, give it the hue the same row has in the dark cabin.
amp = band.max(2)
# one hue per band, taken where that band is strongest on the dark interior (per-row hues were unstable on the
# faint edge rows and painted yellow-green segments into the layer)
row_col = np.zeros((H, 3), np.float32)
band_hues = {}
for name, (ra, rb_) in {"top": (R0, 841), "mid": (841, 946), "low": (946, R1 + 1)}.items():
    ref = band_A[ra:rb_, 1010:1400].reshape(-1, 3)
    w_ = ref.max(1)
    sel = w_ > np.percentile(w_, 90)
    hue = (ref[sel] * w_[sel, None]).sum(0) / (w_[sel].sum() + 1e-3)
    hue = hue / max(hue.max(), 1e-3)
    row_col[ra:rb_] = hue
    band_hues[name] = [round(float(v), 3) for v in hue]
print("band hues (B,G,R):", band_hues)
row_col[:R0] = row_col[R0]; row_col[R1 + 1:] = row_col[R1]
row_col = cv2.GaussianBlur(row_col.reshape(H, 1, 3), (0, 0), sigmaX=0.1, sigmaY=3).reshape(H, 3)
recol = amp[..., None] * row_col[:, None, :]
band_flow = band * (1 - w_g[..., None]) + recol * w_g[..., None]
# The hottest cores were painted past white: red is clipped at 255 in the still, so the lifted light has too
# little red there and turns yellow-green once it is dimmed or laid over the teal cabin. Give red back from green
# using the band's own hue. Values above 255 are fine here: the layer is light, it is clipped after compositing.
clipped_r = cv2.GaussianBlur((f[..., 2] >= 250).astype(np.float32), (0, 0), 1.0)
g_over_r = np.maximum(row_col[:, 1], 0.3)[:, None]
r_true = band_flow[..., 1] / g_over_r
band_flow[..., 2] = band_flow[..., 2] * (1 - clipped_r) + np.maximum(band_flow[..., 2], r_true) * clipped_r
print("red restored on", int((clipped_r > 0.5).sum()), "px; layer max R", round(float(band_flow[..., 2].max()), 0))
bmax = band.max(2)
bands = (bmax > 5).astype(np.uint8) * 255
bands = cv2.dilate(cv2.morphologyEx(bands, cv2.MORPH_OPEN, np.ones((1, 15), np.uint8)), ellipse(7))

# three bands: valleys of the row profile, measured where nothing else is bright (dark cabin interior)
prof = bmax[:, 1040:1380].mean(1)
on = prof > 2.0
runs, s = [], None
for y in range(H):
    if on[y] and s is None:
        s = y
    if (not on[y]) and s is not None:
        runs.append((s, y - 1)); s = None
runs = [(a, c) for a, c in runs if c - a >= 3]
print("band runs in the cabin interior (rows):", runs)
# split into three bands at the two darkest rows between them (profile over the whole width, smoothed)
pw = cv2.GaussianBlur(bmax.mean(1).reshape(-1, 1), (0, 0), 3).ravel()
cut1 = 834 + int(np.argmin(pw[834:862]))
cut2 = 925 + int(np.argmin(pw[925:968]))
band_split = dict(top=[BAND_ROWS[0], cut1], mid=[cut1, cut2], low=[cut2, BAND_ROWS[1]])
print("band split rows:", band_split, "profile at cuts", round(float(pw[cut1]), 1), round(float(pw[cut2]), 1),
      "peaks", round(float(pw[820:834].max()), 1), round(float(pw[cut1:cut2].max()), 1), round(float(pw[cut2:1040].max()), 1))

# ---------------------------------------------------------------- 5. body reflections (measured on the plate)
pb = cv2.GaussianBlur(plate, (0, 0), 1.2)
p_rb, p_lum = pb[..., 2] - pb[..., 0], pb.mean(2)
near_glass = cv2.dilate(glass, ellipse(5))
refl = ((p_rb > 12) & (p_lum > 30) & (near_glass == 0)).astype(np.uint8) * 255
refl = cv2.morphologyEx(refl, cv2.MORPH_OPEN, ellipse(3))
refl = cv2.dilate(refl, ellipse(5))

# ---------------------------------------------------------------- save
def save(name, m):
    cv2.imwrite(os.path.join(MD, name), m)

save("glass.png", glass); save("girl.png", girl); save("curtains.png", curt_all)
for k, m in curtains.items():
    save(f"curtain_{k}.png", m)
save("bands.png", bands); save("body_refl.png", refl)
save("plate.png", np.round(plate).astype(np.uint8))
save("band_layer.png", np.round(np.clip(band_flow, 0, 255)).astype(np.uint8))
np.save(os.path.join(MD, "band_layer_f32.npy"), band_flow.astype(np.float32))
np.save(os.path.join(MD, "plate_f32.npy"), plate.astype(np.float32))
info = dict(size=[W, H], band_rows=BAND_ROWS, windows=win_info, band_runs_interior=runs, band_split=band_split, kv_girl=KV_GIRL, kv_big=KV_BIG, kh=KH, kh_long=KH_LONG,
            girl_poly=GIRL_POLY, curtain_polys=CURTAIN_POLYS,
            area_px=dict(glass=int((glass > 0).sum()), girl=int((girl > 0).sum()), curtains=int((curt_all > 0).sum()),
                         bands=int((bands > 0).sum()), body_refl=int((refl > 0).sum())))
json.dump(info, open(os.path.join(MD, "masks.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(info["windows"]), info["area_px"])

# ---------------------------------------------------------------- schematic
COL = {"glass": (194, 144, 74), "curtain": (74, 195, 139), "girl": (177, 143, 244), "band": (61, 138, 255),
       "refl": (79, 213, 255)}                       # BGR
base = (f * 0.55).astype(np.float32)
vis = base.copy()

def paint(mask, col, a):
    global vis
    m = (mask > 0)[..., None].astype(np.float32) * a
    vis = vis * (1 - m) + np.array(col, np.float32) * m

paint(glass, COL["glass"], 0.38)
paint(curt_all, COL["curtain"], 0.50)
paint(girl, COL["girl"], 0.50)
paint(bands, COL["band"], 0.62)
paint(refl, COL["refl"], 0.85)
vis = np.clip(vis, 0, 255).astype(np.uint8)
for mask, col in ((glass, COL["glass"]), (curt_all, COL["curtain"]), (girl, COL["girl"])):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cs, -1, col, 2, cv2.LINE_AA)
save("schematic_plain.png", vis)

FONT = next((p for p in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
                         "/System/Library/Fonts/STHeiti Medium.ttc") if os.path.exists(p)), None)
if FONT is None:        # the labelled review sheets need a CJK font; every mask above is already written
    print("done (no CJK font found: schematic.jpg and peel_check.jpg skipped; point FONT at a font file to get them)")
    sys.exit(0)
LEG = 250
canvas = Image.new("RGB", (W, H + LEG), (9, 14, 24))
canvas.paste(Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)), (0, 0))
dr = ImageDraw.Draw(canvas)
ft, fs = ImageFont.truetype(FONT, 40), ImageFont.truetype(FONT, 30)
items = [("glass", "车窗玻璃", "车内光线变化的范围"), ("curtain", "窗帘（4 块）", "越往下摆得越大"),
         ("girl", "文静", "越靠头部摆得越大"), ("band", "光带", "剥下来，横向流动"),
         ("refl", "车体反光", "位置不动，只做明灭")]
x = 60
for key, title, note in items:
    c = COL[key][::-1]
    dr.rounded_rectangle([x, H + 52, x + 44, H + 96], 8, fill=c)
    dr.text((x + 62, H + 46), title, font=ft, fill=(240, 236, 226))
    dr.text((x + 62, H + 108), note, font=fs, fill=(150, 160, 176))
    x += 396
dr.text((60, H + 180), "遮罩以外的像素保持原图不动。", font=fs, fill=(150, 160, 176))
canvas.save(os.path.join(MD, "schematic.jpg"), quality=93)

# peel check: rows around the bands, original / plate / bands only
y0, y1 = 760, 1070
strips = [im[y0:y1], np.round(plate[y0:y1]).astype(np.uint8), np.round(np.clip(band_flow[y0:y1] * 1.0, 0, 255)).astype(np.uint8)]
gap = np.full((8, W, 3), 60, np.uint8)
pc = np.vstack([strips[0], gap, strips[1], gap, strips[2]])
pil = Image.fromarray(cv2.cvtColor(pc, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
for i, t in enumerate(("原图", "光带下面", "只有光带")):
    yy = i * (y1 - y0 + 8) + 8
    dr.rectangle([8, yy, 8 + 36 * len(t) + 24, yy + 50], fill=(0, 0, 0))
    dr.text((20, yy + 4), t, font=ImageFont.truetype(FONT, 34), fill=(255, 220, 120))
pil.save(os.path.join(MD, "peel_check.jpg"), quality=93)
x0, x1, y0, y1 = 560, 1040, 780, 1050
z = np.hstack([im[y0:y1, x0:x1], np.round(plate[y0:y1, x0:x1]).astype(np.uint8),
               np.round(np.clip(band_flow[y0:y1, x0:x1] * 2.0, 0, 255)).astype(np.uint8)])
cv2.imwrite(os.path.join(MD, "peel_check_girl.png"), cv2.resize(z, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC))
print("done")
