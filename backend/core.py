"""Core image analysis pipeline shared by Gradio and FastAPI backends."""
import base64
import glob
import io
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules
from sklearn.cluster import KMeans

CFG = "segformer_mit-b0_8xb2-160k_ade20k-512x512.py"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
STRICT = {"building", "house", "skyscraper", "garage", "roof", "windowpane", "door", "balcony"}
KL, KB, MORPH_KERNEL = 1.0, 0.5, 3
TOPK, PALETTE_W, WHITE_TH, BLACK_TH, MIN_SAMPLES = 5, 120, 240, 20, 500

_model = None
_building_ids = None
_ckpt = None


def load_model():
    """Load segmentation model once and cache building class ids."""
    global _model, _building_ids, _ckpt
    if _model is None:
        register_all_modules()
        cands = sorted(glob.glob("segformer_mit-b0_*ade20k*.pth"))
        if not cands:
            raise RuntimeError("Checkpoint missing; please download the ADE20K weight file first.")
        _ckpt = cands[0]
        _model = init_model(CFG, _ckpt, device=DEVICE)
        classes = _model.dataset_meta.get("classes", [])
        name2id = {name: i for i, name in enumerate(classes)}
        ids = [name2id[name] for name in STRICT if name in name2id]
        if "building" in name2id and name2id["building"] not in ids:
            ids.append(name2id["building"])
        _building_ids = sorted(set(ids))
    return _model, _building_ids


def shadow_mask_lab(img_bgr: np.ndarray, valid_mask255: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, B = lab[..., 0], lab[..., 2]
    m = valid_mask255 == 255
    if not np.any(m):
        return np.zeros_like(L, np.uint8)
    Lm, Bm = L[m], B[m]
    Lm_mean, Lm_std = float(Lm.mean()), float(Lm.std() + 1e-6)
    Bm_mean, Bm_std = float(Bm.mean()), float(Bm.std() + 1e-6)
    s = ((L < (Lm_mean - KL * Lm_std)) & (B < (Bm_mean - KB * Bm_std)) & m).astype(np.uint8) * 255
    if MORPH_KERNEL > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        s = cv2.morphologyEx(s, cv2.MORPH_OPEN, k, iterations=1)
    return s


def get_palette_from_bgra(bgra: np.ndarray) -> List[Tuple[List[int], float]]:
    bgr, alpha = bgra[..., :3], bgra[..., 3]
    mask = alpha > 0
    if mask.sum() < MIN_SAMPLES:
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sel = rgb[mask].astype(np.uint8)
    keep = ~((sel >= WHITE_TH).all(axis=1) | (sel <= BLACK_TH).all(axis=1))
    sel = sel[keep]
    if sel.shape[0] < MIN_SAMPLES:
        return []
    uniq = np.unique(sel, axis=0)
    n = int(min(TOPK, max(1, len(uniq))))
    km = KMeans(n_clusters=n, n_init="auto", random_state=42).fit(sel.astype(np.float32))
    centers = km.cluster_centers_.clip(0, 255).astype(np.uint8)
    counts = np.bincount(km.labels_, minlength=n).astype(float)
    ratios = counts / counts.sum()
    order = np.argsort(-ratios)
    return [(centers[i].tolist(), float(ratios[i])) for i in order]


def compose_with_palette_keep_alpha(bgra: np.ndarray, colors: List[Tuple[List[int], float]]) -> np.ndarray:
    h = bgra.shape[0]
    card = np.zeros((h, PALETTE_W, 4), np.uint8)
    card[..., 3] = 255
    if colors:
        y = 0
        for rgb, ratio in colors:
            bh = max(1, int(round(ratio * h)))
            card[y:y + bh] = (rgb[2], rgb[1], rgb[0], 255)
            y += bh
        if y < h:
            card[y:h] = card[y - 1] if y > 0 else (60, 60, 60, 255)
    return np.concatenate([bgra, card], axis=1)


def analyze_image_array(img_bgr: np.ndarray) -> Tuple[np.ndarray, List[Tuple[List[int], float]]]:
    model, building_ids = load_model()
    if model is None or not building_ids:
        raise RuntimeError("Model not loaded; please check checkpoint availability.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = inference_model(model, img_rgb)
    seg = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)
    mask255 = (np.isin(seg, building_ids)).astype(np.uint8) * 255

    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[mask255 == 0, 3] = 0
    sh = shadow_mask_lab(img_bgr, mask255)
    bgra[sh == 255, 3] = 0

    colors = get_palette_from_bgra(bgra)
    out = compose_with_palette_keep_alpha(bgra, colors)
    return out, colors


def analyze_image_pil(pil_img: Image.Image) -> Tuple[np.ndarray, List[Tuple[List[int], float]]]:
    img_rgb = pil_img.convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    return analyze_image_array(img_bgr)


def analyze_image_bytes(data: bytes) -> Tuple[np.ndarray, List[Tuple[List[int], float]]]:
    try:
        pil_img = Image.open(io.BytesIO(data))
    except Exception as exc:  # pillow cannot open
        raise ValueError("Invalid image data") from exc
    return analyze_image_pil(pil_img)


def encode_png(bgra: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return buf.tobytes()


def to_base64_png(bgra: np.ndarray) -> str:
    png_bytes = encode_png(bgra)
    return base64.b64encode(png_bytes).decode("utf-8")