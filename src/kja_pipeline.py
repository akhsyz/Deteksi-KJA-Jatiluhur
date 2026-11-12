"""
kja_pipeline.py — Deteksi dan estimasi jumlah unit Keramba Jaring Apung (KJA)
dari citra satelit top-view dengan metode klasik (tanpa deep learning).

Alur: preprocessing -> fitur tekstur GLCM + fitur hijau relatif + kerapatan garis rangka
      -> K-Means -> post-processing morfologi -> luas & estimasi jumlah unit -> evaluasi.
"""
import time
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

GLCM_PROPS = ["contrast", "homogeneity", "energy", "entropy"]


# ------------------------------------------------------------------ 1. citra & preprocessing
def load_image(path, crop_bottom=0):
    """Baca citra BGR. crop_bottom = jumlah baris bawah yang dibuang (logo Google Earth / garis skala)."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr[:-crop_bottom] if crop_bottom else bgr


def preprocess(bgr, median_ksize=3, clahe_clip=2.0, clahe_tile=8):
    """Grayscale -> median filter (redam noise) -> CLAHE (naikkan kontras lokal)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    den = cv2.medianBlur(gray, median_ksize)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    return clahe.apply(den)


# ------------------------------------------------------------------ 2. fitur
def glcm_maps(enh, win=15, step=4, levels=32, distance=1, verbose=True):
    """Peta fitur GLCM dengan jendela geser. Dihitung tiap `step` piksel lalu diperbesar ke ukuran citra."""
    h, w = enh.shape
    q = (enh.astype(np.float32) / 256 * levels).astype(np.uint8)
    pad = win // 2
    qp = cv2.copyMakeBorder(q, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    ys, xs = np.arange(0, h, step), np.arange(0, w, step)
    out = {p: np.zeros((len(ys), len(xs)), np.float32) for p in GLCM_PROPS}
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    t = time.time()
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            G = graycomatrix(qp[y:y + win, x:x + win], [distance], angles,
                             levels=levels, symmetric=True, normed=True)
            for p in ("contrast", "homogeneity", "energy"):
                out[p][i, j] = graycoprops(G, p).mean()
            P = G.mean(axis=3)[:, :, 0]
            P = P[P > 0]
            out["entropy"][i, j] = -(P * np.log2(P)).sum()
        if verbose and (i % max(1, len(ys) // 10) == 0):
            print(f"  GLCM baris {i + 1}/{len(ys)} ({time.time() - t:.0f} detik)")
    if verbose:
        print(f"  GLCM selesai: {len(ys) * len(xs)} jendela, {time.time() - t:.0f} detik")
    return {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) for k, v in out.items()}


def green_relative(bgr, win=15):
    """Excess Green (2G-R-B) rata-rata lokal, dikurangi latar sekitarnya.
    Dikurangi latar supaya beda warna antar-citra satelit (mosaik) tidak dianggap tanaman."""
    h, w = bgr.shape[:2]
    f = bgr.astype(np.float32) / 255
    exg = 2 * f[:, :, 1] - f[:, :, 2] - f[:, :, 0]
    local = cv2.blur(exg, (win, win))
    small = cv2.resize(exg, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    u8 = (np.clip(small + 1, 0, 2) * 127).astype(np.uint8)
    bg = cv2.resize(cv2.medianBlur(u8, 31), (w, h)).astype(np.float32) / 127 - 1
    return local - bg


def line_density(bgr, enh, win=15, tophat_size=5, max_green=0.05):
    """Kerapatan garis rangka: struktur terang yang lebih tipis dari `tophat_size` piksel
    (white top-hat + ambang Otsu) dan tidak hijau, dirata-ratakan dalam jendela."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_size, tophat_size))
    th = cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, k)
    t, _ = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    f = bgr.astype(np.float32) / 255
    exg = 2 * f[:, :, 1] - f[:, :, 2] - f[:, :, 0]
    lines = ((th > t) & (exg < max_green)).astype(np.float32)
    return cv2.blur(lines, (win, win))


# ------------------------------------------------------------------ 3. segmentasi
def _smooth_labels(lbl, k, size=15):
    votes = np.stack([cv2.blur((lbl == c).astype(np.float32), (size, size)) for c in range(k)], -1)
    return votes.argmax(-1)


def segment_kmeans(features, k=3, kja_by="line_density", sample=200000, seed=0):
    """K-Means pada tumpukan fitur (dict nama -> peta 2D).
    Klaster KJA = klaster dengan rata-rata fitur `kja_by` tertinggi."""
    names = list(features)
    h, w = features[names[0]].shape
    X = np.stack([features[n] for n in names], -1).reshape(-1, len(names))
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), min(sample, len(Xs)), replace=False)
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Xs[idx])
    labels = _smooth_labels(km.predict(Xs).reshape(h, w), k)
    ref = features[kja_by]
    means = [ref[labels == c].mean() if (labels == c).any() else -np.inf for c in range(k)]
    kja_cluster = int(np.argmax(means))
    summary = []
    for c in range(k):
        m = labels == c
        summary.append({"klaster": c, "persen": round(m.mean() * 100, 1),
                        **{n: round(float(features[n][m].mean()), 3) for n in names}})
    return labels, kja_cluster, summary


def select_kja_clusters(summary, feature="line_density", frac=0.4):
    """Klaster dianggap KJA bila rata-rata kerapatan garisnya >= frac x klaster tertinggi.
    Perlu karena baris keramba yang garisnya samar kadang dipisah K-Means ke klaster sendiri."""
    vals = [s[feature] for s in summary]
    top = max(vals)
    return [s["klaster"] for s in summary if s[feature] >= frac * top]


def line_gate(line_dens, win=31, threshold=0.06):
    """Syarat tambahan: kerapatan garis dalam jendela lebar (±27 m) harus > threshold.
    Membuang hamparan tanaman yang bertekstur tetapi tidak punya garis rangka."""
    return cv2.blur(line_dens, (win, win)) > threshold


def raw_lines(enh, tophat_size=5):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_size, tophat_size))
    th = cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, k)
    t, _ = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th > t


def refine_to_frame(mask, enh, close_px=13):
    """Pangkas tepi: garis rangka di dalam mask ditutup (closing) menjadi bidang keramba,
    lalu diiriskan dengan mask. Hasilnya batas KJA menempel ke rangka, tidak melebar ke air."""
    lines = (raw_lines(enh) & mask).astype(np.uint8)
    hull = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)))
    hull = cv2.morphologyEx(hull, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return hull.astype(bool) & mask


def color_baseline(bgr):
    """Pembanding: K-Means k=2 hanya dengan warna (Lab). Klaster lebih terang = 'KJA'."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(lab), min(200000, len(lab)), replace=False)
    km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(lab[idx])
    lbl = km.predict(lab)
    bright = int(np.argmax(km.cluster_centers_[:, 0]))
    return (lbl == bright).reshape(bgr.shape[:2])


def postprocess(mask, m_per_px, close_px=7, open_px=5, min_area_m2=150, exclude=None):
    """Closing (tutup celah) -> opening (buang bintik) -> buang objek < min_area_m2.
    exclude = mask boolean area yang dikecualikan (mis. daratan), opsional."""
    m = mask.astype(np.uint8) * 255
    if exclude is not None:
        m[exclude] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    min_px = min_area_m2 / (m_per_px ** 2)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lbl]


def polygon_mask(shape, polygons):
    """Mask dari daftar poligon [[(x,y),...], ...], untuk masker daratan manual."""
    m = np.zeros(shape[:2], np.uint8)
    for poly in polygons:
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m.astype(bool)


# ------------------------------------------------------------------ 4. luas & estimasi
def area_and_units(mask, m_per_px, unit_area_m2=49.0):
    area_m2 = float(mask.sum()) * m_per_px ** 2
    return {"luas_m2": round(area_m2, 1), "luas_ha": round(area_m2 / 1e4, 3),
            "luas_per_unit_m2": round(unit_area_m2, 2),
            "estimasi_unit": int(round(area_m2 / unit_area_m2))}


# ------------------------------------------------------------------ 5. kunci jawaban & evaluasi
def read_answer_key(path, shape=None):
    """Baca kunci jawaban PNG: piksel merah (R tinggi, G & B rendah, tidak transparan) = KJA."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    b, g, r, a = [img[:, :, i].astype(int) for i in range(4)]
    m = (r > 150) & (g < 100) & (b < 100) & (a > 127)
    if shape is not None and m.shape != tuple(shape[:2]):
        raise ValueError(f"Ukuran kunci jawaban {m.shape} beda dengan citra {tuple(shape[:2])}")
    return m


def evaluate(pred, gt):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / pred.size
    return {"IoU": round(iou, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "F1": round(f1, 4), "akurasi_piksel": round(acc, 4)}


def error_map(rgb, pred, gt):
    """Hijau = benar (TP), merah = salah tandai (FP), biru = terlewat (FN)."""
    o = rgb.astype(np.float32).copy()
    for m, col in [(pred & gt, (40, 200, 90)), (pred & ~gt, (230, 40, 40)), (~pred & gt, (40, 120, 255))]:
        o[m] = o[m] * 0.35 + np.array(col) * 0.65
    return o.astype(np.uint8)


def answer_key_rows(gt):
    """Pisahkan kunci jawaban jadi baris-baris keramba (komponen terhubung) untuk kalibrasi."""
    n, lbl, st, cent = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    rows = [{"id": i, "luas_px": int(st[i, cv2.CC_STAT_AREA]), "pusat": (int(cent[i][0]), int(cent[i][1]))}
            for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 30]
    return lbl, rows


def calibrate_unit_area(gt_lbl, counts, m_per_px):
    """counts = {id_komponen: jumlah_kotak}. Hasil: rata-rata luas satu kotak (m²)."""
    per = []
    for cid, n in counts.items():
        px = int((gt_lbl == cid).sum())
        if n > 0 and px > 0:
            per.append(px * m_per_px ** 2 / n)
    if not per:
        raise ValueError("counts kosong atau id tidak ditemukan")
    return float(np.mean(per)), float(np.std(per)), per


def overlay(rgb, mask, color=(255, 122, 40), alpha=0.5):
    o = rgb.astype(np.float32).copy()
    o[mask] = o[mask] * (1 - alpha) + np.array(color) * alpha
    return o.astype(np.uint8)
