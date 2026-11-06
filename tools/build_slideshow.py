"""Snaze slideshow builder (house style).
Usage: python3 build_slideshow.py spec.json --photos DIR --out DIR [--screenshot PNG]
See the snaze-slideshow skill for the spec format."""
import sys, os, json, argparse, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

W, H = 1080, 1920
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = None
PHOTOS = None
SEMI = os.path.join(BASE, "fonts/Sora2-SemiBold.ttf")
REG = os.path.join(BASE, "fonts/Sora2-Regular.ttf")
PF_BOLD = os.path.join(BASE, "fonts/PlayfairDisplay-Bold.ttf")
ORANGE = (238, 97, 52)
CREAM = (250, 247, 242)

# ---------- helpers ----------
def font(path, size): return ImageFont.truetype(path, size)

def text_w(f, s): return f.getbbox(s)[2] - f.getbbox(s)[0]

def greedy(words, f, maxw):
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if text_w(f, t) <= maxw or not cur: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def balanced_wrap(text, f, maxw):
    words = text.split()
    best = None
    for frac in np.linspace(1.0, 0.55, 19):
        lines = greedy(words, f, maxw * frac)
        if any(text_w(f, l) > maxw for l in lines): continue
        widths = [text_w(f, l) for l in lines]
        key = (len(lines), float(np.var(widths)))
        if best is None or key < best[0]: best = (key, lines)
    return best[1]

def fit_size(text, path, start, minimum, maxw, max_lines):
    s = start
    while s > minimum:
        f = font(path, s)
        if len(balanced_wrap(text, f, maxw)) <= max_lines: return f
        s -= 2
    return font(path, minimum)

def vignette(img, strength=120):
    w, h = img.size
    y, x = np.ogrid[:h, :w]
    d = np.sqrt(((x - w/2)/(w/2))**2 + ((y - h/2)/(h/2))**2)
    mask = np.clip((d - 0.55) / 0.75, 0, 1) ** 1.6 * strength
    m = Image.fromarray(mask.astype(np.uint8), "L")
    black = Image.new("RGB", img.size, (0, 0, 0))
    return Image.composite(black, img, m)

def cover_crop(img, w, h):
    r = max(w / img.width, h / img.height)
    img = img.resize((int(img.width*r)+1, int(img.height*r)+1), Image.LANCZOS)
    l, t = (img.width - w)//2, (img.height - h)//2
    return img.crop((l, t, l+w, t+h))

def photo_bg(path, seed=1, target_mean=78):
    img = cover_crop(Image.open(path).convert("RGB"), W, H)
    mean = np.asarray(img.convert("L")).mean()
    img = ImageEnhance.Brightness(img).enhance(float(np.clip(target_mean/mean, 0.5, 1.4)))
    img = ImageEnhance.Contrast(img).enhance(1.22)
    img = ImageEnhance.Color(img).enhance(0.92)
    a = np.asarray(img).astype(np.float32) / 255.0
    a = a ** 1.35
    lum = a.mean(axis=2, keepdims=True)
    a += (1 - lum) * np.array([0.04, 0.05, 0.07]) * 0.6
    a += lum * np.array([0.06, 0.02, -0.01]) * 0.6
    # house grade: pull every photo into the same warm-dark family as the CTA slide
    a = np.clip(a, 0, 1)
    lum = (a * np.array([0.299, 0.587, 0.114])).sum(axis=2, keepdims=True)
    a = lum + (a - lum) * 0.62                      # shared saturation level
    dark, light = np.array([14, 11, 12]) / 255, np.array([255, 226, 200]) / 255
    duo = dark + (light - dark) * lum
    a = a * 0.58 + duo * 0.42                       # shared warm tint
    a = np.minimum(a, 0.86 + a * 0.0)               # tame hot highlights (lamps, paper)
    a = a * 255 + np.random.default_rng(seed).normal(0, 3.0, a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return vignette(img, 120)

def safety_scrim(img, top, bottom, max_alpha=135):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rectangle([120, top - 40, W - 120, bottom + 40], fill=max_alpha)
    m = m.filter(ImageFilter.GaussianBlur(80))
    return Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), img, m)

def soft_text(img, xy, text, f, fill, shadow_alpha=110, anchor="la", feats=None):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((xy[0], xy[1] + 3), text, font=f, fill=(0, 0, 0, shadow_alpha), anchor=anchor, features=feats)
    layer = layer.filter(ImageFilter.GaussianBlur(2))
    img.paste(layer, (0, 0), layer)
    ImageDraw.Draw(img).text(xy, text, font=f, fill=fill, anchor=anchor, features=feats)

def footer(img):
    f = font(SEMI, 24)
    soft_text(img, (W//2, H - 90), "@snazeapp", f, (235, 230, 224), 120, anchor="mm")

# ---------- photo slides ----------
def compose_slide(name, photo, head, sub, seed):
    img = photo_bg(os.path.join(PHOTOS, photo), seed)
    maxw = W - 400
    hf = fit_size(head.replace('\n',' '), SEMI, 50, 36, maxw, 3) if '\n' not in head else font(SEMI, 48)
    sf = font(REG, 33)
    hl = head.split('\n') if '\n' in head else balanced_wrap(head, hf, maxw)
    sl = balanced_wrap(sub, sf, min(maxw, 700))
    hlh, slh, gap = int(hf.size * 1.25), int(sf.size * 1.5), 34
    block = len(hl)*hlh + gap + len(sl)*slh
    text_bottom = int(H * 0.54)
    top = text_bottom - block
    img = safety_scrim(img, top, text_bottom)
    y = top
    for i, l in enumerate(hl):
        num, rest = (l.split(' ', 1) + [''])[:2] if (i == 0 and l[:1].isdigit() and l.split(' ')[0].rstrip('.').isdigit()) else ('', l)
        if num:
            nf = font(PF_BOLD, int(hf.size * 1.32))
            nw, sp = nf.getlength(num, features=["lnum"]), int(hf.size * 0.32)
            x0 = int(W//2 - (nw + sp + text_w(hf, rest))//2)
            base = y + hf.getmetrics()[0]
            soft_text(img, (x0, base), num, nf, ORANGE, 120, anchor="ls", feats=["lnum"])
            soft_text(img, (int(x0 + nw + sp), base), rest, hf, CREAM, 120, anchor="ls")
        else:
            soft_text(img, (W//2, y), l, hf, CREAM, 120, anchor="ma")
        y += hlh
    y += gap
    for l in sl:
        soft_text(img, (W//2, y), l, sf, (240, 234, 226), 120, anchor="ma"); y += slh
    footer(img)
    img.save(os.path.join(OUT, name + ".png"))


def compose_cover(name, photo, number, lines, sub, seed=1):
    img = photo_bg(os.path.join(PHOTOS, photo), seed)
    big = font(PF_BOLD, 360)
    lf = font(SEMI, 52)
    sf = font(REG, 33)
    sl = balanced_wrap(sub, sf, 640)
    bb = big.getbbox(number, features=["lnum"])
    big_h = bb[3] - bb[1]
    block = big_h + 46 + 66 * len(lines) + 4 + 40 + len(sl) * 48
    text_bottom = int(H * 0.56)
    top = text_bottom - block
    img = safety_scrim(img, top - 20, text_bottom, 150)
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).text((W//2, top), number, font=big, fill=150, anchor="mt", features=["lnum"])
    img.paste(Image.new("RGB", (W, H), (200, 80, 40)), (0, 0), glow.filter(ImageFilter.GaussianBlur(40)))
    soft_text(img, (W//2, top), number, big, ORANGE, 140, anchor="mt", feats=["lnum"])
    y = top + big_h + 46
    for l in lines:
        soft_text(img, (W//2, y), l, lf, CREAM, 120, anchor="ma"); y += 66
    y += 44
    for l in sl:
        soft_text(img, (W//2, y), l, sf, (240, 234, 226), 120, anchor="ma"); y += 48
    footer(img)
    img.save(os.path.join(OUT, name + ".png"))


def segments(line, hf, nf):
    """Split 'You train {5x} a week.' into (text, font, color, features) runs; {..} = orange Playfair."""
    out = []
    for part in re.split(r"(\{[^}]*\})", line):
        if not part: continue
        if part.startswith("{"):
            out.append((part[1:-1], nf, ORANGE, ["lnum"]))
        else:
            out.append((part, hf, CREAM, None))
    return out

def compose_cover_b(name, photo, lines, sub, seed=1):
    img = photo_bg(os.path.join(PHOTOS, photo), seed)
    hf = font(SEMI, 60)
    nf = font(PF_BOLD, 84)
    sf = font(REG, 34)
    lh = 84
    sl = balanced_wrap(sub, sf, 700)
    block = len(lines) * lh + 40 + len(sl) * 50
    text_bottom = int(H * 0.54)
    top = text_bottom - block
    img = safety_scrim(img, top - 20, text_bottom, 150)
    base = top + hf.getmetrics()[0] + 10
    for line in lines:
        segs = segments(line, hf, nf)
        total = sum(f.getlength(t, features=ft) for t, f, _, ft in segs)
        x = W/2 - total/2
        for t, f, c, ft in segs:
            soft_text(img, (int(x), base), t, f, c, 120, anchor="ls", feats=ft); x += f.getlength(t, features=ft)
        base += lh
    y = top + len(lines) * lh + 40
    for l in sl:
        soft_text(img, (W//2, y), l, sf, (240, 234, 226), 120, anchor="ma"); y += 50
    footer(img)
    img.save(os.path.join(OUT, name + ".png"))

# ---------- closing slide ----------
def diagonal_gradient(w, h, c1, c2):
    x = np.linspace(0, 1, w)[None, :]; y = np.linspace(0, 1, h)[:, None]
    t = np.clip((x + y) / 2, 0, 1)[..., None]
    a = np.array(c1) * (1 - t) + np.array(c2) * t
    return Image.fromarray(a.astype(np.uint8))

def rounded_mask(size, r):
    m = Image.new("L", size, 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0]-1, size[1]-1], r, fill=255); return m

def build_phone(shot, phone_w=560):
    sw = phone_w
    bezel = int(0.042 * sw)
    screen_w = sw - 2 * bezel
    screen_h = int(screen_w * shot.height / shot.width)
    ph = screen_h + 2 * bezel
    r_out, r_in = int(sw * 0.16), int(sw * 0.16) - bezel
    pad = 30
    canvas = Image.new("RGBA", (sw + 2*pad, ph + 2*pad), (0, 0, 0, 0))
    # side buttons
    d = ImageDraw.Draw(canvas)
    btn = (70, 72, 78, 255)
    d.rounded_rectangle([pad + sw - 4, pad + int(ph*0.24), pad + sw + 7, pad + int(ph*0.36)], 4, fill=btn)
    for a, b in [(0.17, 0.21), (0.25, 0.32), (0.34, 0.41)]:
        d.rounded_rectangle([pad - 7, pad + int(ph*a), pad + 4, pad + int(ph*b)], 4, fill=btn)
    # titanium frame
    frame = diagonal_gradient(sw, ph, (178, 180, 186), (52, 54, 60)).convert("RGBA")
    streak = np.random.default_rng(3).normal(0, 6, (ph, 1)).repeat(sw, 1)
    fa = np.asarray(frame).astype(np.float32); fa[..., :3] += streak[..., None]
    frame = Image.fromarray(np.clip(fa, 0, 255).astype(np.uint8))
    canvas.paste(frame, (pad, pad), rounded_mask((sw, ph), r_out))
    # inner black rim
    rim = int(bezel * 0.55)
    blk = Image.new("RGBA", (sw - 2*rim, ph - 2*rim), (8, 8, 10, 255))
    canvas.paste(blk, (pad + rim, pad + rim), rounded_mask(blk.size, r_out - rim))
    # screen
    scr = shot.convert("RGBA").resize((screen_w, screen_h), Image.LANCZOS)
    canvas.paste(scr, (pad + bezel, pad + bezel), rounded_mask((screen_w, screen_h), r_in))
    # dynamic island
    iw, ih = int(screen_w * 0.30), int(screen_w * 0.085)
    ix, iy = pad + sw//2 - iw//2, pad + bezel + int(screen_w * 0.03)
    d = ImageDraw.Draw(canvas)
    # specular rim lines
    d.rounded_rectangle([pad, pad, pad + sw - 1, pad + ph - 1], r_out, outline=(225, 226, 230, 200), width=2)
    d.rounded_rectangle([pad + 3, pad + 3, pad + sw - 4, pad + ph - 4], r_out - 3, outline=(30, 30, 34, 180), width=2)
    # glass reflection
    gl = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(gl).polygon([(pad + bezel, pad + bezel), (pad + int(sw*0.55), pad + bezel),
                                (pad + bezel, pad + int(ph*0.42))], fill=28)
    gl = gl.filter(ImageFilter.GaussianBlur(30))
    sm = Image.new("L", canvas.size, 0); sm.paste(rounded_mask((screen_w, screen_h), r_in), (pad + bezel, pad + bezel))
    gl = Image.fromarray(np.minimum(np.asarray(gl), np.asarray(sm)))
    canvas.paste(Image.new("RGBA", canvas.size, (255, 255, 255, 255)), (0, 0), gl)
    return canvas

def compose_closing(screenshot_path, c):
    a = np.linspace(0, 1, H)[:, None, None]
    bg = np.array([44, 30, 30]) * (1 - a) + np.array([14, 12, 13]) * a
    bg = bg.repeat(W, 1) + np.random.default_rng(9).normal(0, 2.5, (H, W, 3))
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGB")
    maxw = W - 360
    y = 150
    soft_text(img, (W//2, y), c["kicker"], font(SEMI, 24), ORANGE, 90, anchor="ma"); y += 62
    hf = font(SEMI, 60)
    soft_text(img, (W//2, y), c["headline"], hf, CREAM, 120, anchor="ma"); y += 96
    sf = font(REG, 31)
    for l in balanced_wrap(c["sub"], sf, maxw):
        soft_text(img, (W//2, y), l, sf, (222, 214, 206), 100, anchor="ma"); y += 46
    band_top = y
    # phone
    phone = build_phone(Image.open(screenshot_path), phone_w=660).rotate(-7, resample=Image.BICUBIC, expand=True)
    phone_y = H - int(phone.height * 0.80)
    # CTA block centered between band_top and phone_y
    bw, bh = 420, 104
    note_h = 44
    block_h = bh + 18 + note_h
    by = band_top + (phone_y - band_top - block_h) // 2
    bx = W//2 - bw//2
    # glow behind phone
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse([W//2 - 380, phone_y - 60, W//2 + 380, H + 400], fill=60)
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    img.paste(Image.new("RGB", (W, H), (92, 58, 44)), (0, 0), glow)
    # phone shadow + phone
    px = W//2 - phone.width//2
    sh = Image.new("L", (W, H), 0)
    sh.paste(phone.split()[3].point(lambda v: int(v * 0.6)), (px + 24, phone_y + 36))
    sh = sh.filter(ImageFilter.GaussianBlur(34))
    img.paste(Image.new("RGB", (W, H), (0, 0, 0)), (0, 0), sh)
    img.paste(phone, (px, phone_y), phone)
    # button glow + shadow
    g = Image.new("L", (W, H), 0)
    ImageDraw.Draw(g).rounded_rectangle([bx - 18, by - 18, bx + bw + 18, by + bh + 18], bh//2 + 18, fill=85)
    img.paste(Image.new("RGB", (W, H), (240, 110, 55)), (0, 0), g.filter(ImageFilter.GaussianBlur(26)))
    s = Image.new("L", (W, H), 0)
    ImageDraw.Draw(s).rounded_rectangle([bx, by + 12, bx + bw, by + bh + 12], bh//2, fill=140)
    img.paste(Image.new("RGB", (W, H), (0, 0, 0)), (0, 0), s.filter(ImageFilter.GaussianBlur(14)))
    btn = diagonal_gradient(bw, bh, (255, 163, 102), (237, 108, 58))
    hi = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(hi).rounded_rectangle([10, 6, bw - 10, bh//2], bh//4, fill=60)
    btn.paste(Image.new("RGB", (bw, bh), (255, 255, 255)), (0, 0), hi.filter(ImageFilter.GaussianBlur(6)))
    img.paste(btn, (bx, by), rounded_mask((bw, bh), bh//2))
    ImageDraw.Draw(img).text((W//2, by + bh//2 + 2), "Download now", font=font(SEMI, 40), fill=(30, 16, 8), anchor="mm")
    soft_text(img, (W//2, by + bh + 18 + 6), "on the App Store", font(REG, 24), (214, 206, 198), 90, anchor="ma")
    img.save(os.path.join(OUT, "07-cta.png"))

DEFAULT_CLOSING = {"kicker": "no more guessing", "headline": "Meet Snaze",
                   "sub": "Snap a photo of your meal and Snaze tracks your calories and macros for you."}

def main():
    global OUT, PHOTOS
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("--photos", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--screenshot", default=os.path.join(BASE, "..", "assets", "app", "scan-result-poke-bowl.png"))
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    OUT, PHOTOS = args.out, args.photos
    os.makedirs(OUT, exist_ok=True)
    # One hook per slideshow: spec["cover"] = {"type": "listicle"|"question", ...}.
    # (Legacy specs with cover_a/cover_b still build both as 01-cover-A/B.)
    if "cover" in spec:
        c = spec["cover"]
        if c["type"] == "listicle":
            compose_cover("01-cover", spec["cover_photo"], c["number"], c["lines"], c["sub"], seed=1)
        else:
            compose_cover_b("01-cover", spec["cover_photo"], c["lines"], c["sub"], seed=1)
    else:
        a, b = spec["cover_a"], spec["cover_b"]
        compose_cover("01-cover-A", spec["cover_photo"], a["number"], a["lines"], a["sub"], seed=1)
        compose_cover_b("01-cover-B", spec["cover_photo"], b["lines"], b["sub"], seed=1)
    reasons = spec["reasons"]
    assert len(reasons) == 5, "exactly 5 reasons (4 unrelated + 1 pivot)"
    for i, r in enumerate(reasons, start=1):
        compose_slide(f"{i+1:02d}-reason{i}", r["photo"], f"{i}. {r['head']}", r["sub"], seed=i + 1)
    compose_closing(args.screenshot, {**DEFAULT_CLOSING, **spec.get("closing", {})})
    print("done ->", OUT)

if __name__ == "__main__":
    main()
