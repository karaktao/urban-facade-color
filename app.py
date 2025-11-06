import base64, glob
from pathlib import Path
import numpy as np, cv2, gradio as gr
from PIL import Image
from sklearn.cluster import KMeans

import torch
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules

# ---- 配置 ----
CFG  = 'segformer_mit-b0_8xb2-160k_ade20k-512x512.py'
CKPT = None  # 启动时在 startup.sh 下载，并在 load_model 里寻找
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

STRICT = {'building','house','skyscraper','garage','roof','windowpane','door','balcony'}
KL, KB, MORPH_KERNEL = 1.0, 0.5, 3
TOPK, PALETTE_W, WHITE_TH, BLACK_TH, MIN_SAMPLES = 5, 120, 240, 20, 500

_model = None
_building_ids = None

def load_model():
    global _model, _building_ids, CKPT
    if _model is None:
        register_all_modules()
        cands = sorted(glob.glob('segformer_mit-b0_*ade20k*.pth'))
        assert cands, 'checkpoint missing'
        CKPT = cands[0]
        _model = init_model(CFG, CKPT, device=DEVICE)
        classes = _model.dataset_meta.get('classes', [])
        name2id = {n:i for i,n in enumerate(classes)}
        ids = [name2id[n] for n in STRICT if n in name2id]
        if 'building' in name2id and name2id['building'] not in ids:
            ids.append(name2id['building'])
        _building_ids = sorted(set(ids))
    return _model, _building_ids

def shadow_mask_lab(img_bgr, valid_mask255):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, B = lab[...,0], lab[...,2]
    m = valid_mask255==255
    if not np.any(m): return np.zeros_like(L, np.uint8)
    Lm, Bm = L[m], B[m]
    Lm_mean, Lm_std = float(Lm.mean()), float(Lm.std()+1e-6)
    Bm_mean, Bm_std = float(Bm.mean()), float(Bm.std()+1e-6)
    s = ((L < (Lm_mean - KL*Lm_std)) & (B < (Bm_mean - KB*Bm_std)) & m).astype(np.uint8)*255
    if MORPH_KERNEL>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MORPH_KERNEL,MORPH_KERNEL))
        s = cv2.morphologyEx(s, cv2.MORPH_OPEN, k, iterations=1)
    return s

def get_palette_from_bgra(bgra):
    bgr, alpha = bgra[...,:3], bgra[...,3]
    mask = alpha>0
    if mask.sum()<MIN_SAMPLES: return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sel = rgb[mask].astype(np.uint8)
    keep = ~((sel>=WHITE_TH).all(axis=1)|(sel<=BLACK_TH).all(axis=1))
    sel = sel[keep]
    if sel.shape[0]<MIN_SAMPLES: return []
    uniq = np.unique(sel, axis=0)
    n = int(min(TOPK, max(1,len(uniq))))
    km = KMeans(n_clusters=n, n_init="auto", random_state=42).fit(sel.astype(np.float32))
    centers = km.cluster_centers_.clip(0,255).astype(np.uint8)
    counts  = np.bincount(km.labels_, minlength=n).astype(float)
    ratios  = counts/counts.sum()
    order = np.argsort(-ratios)
    return [(centers[i].tolist(), float(ratios[i])) for i in order]

def compose_with_palette_keep_alpha(bgra, colors):
    h = bgra.shape[0]
    card = np.zeros((h, PALETTE_W, 4), np.uint8); card[...,3]=255
    if colors:
        y=0
        for rgb,ratio in colors:
            bh=max(1,int(round(ratio*h)))
            card[y:y+bh] = (rgb[2],rgb[1],rgb[0],255)
            y += bh
        if y<h: card[y:h]=card[y-1] if y>0 else (60,60,60,255)
    return np.concatenate([bgra, card], axis=1)

def pipeline(pil_img: Image.Image):
    model, building_ids = load_model()
    img_rgb = pil_img.convert('RGB')
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

    # 1) 分割
    result = inference_model(model, img_rgb)
    seg = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)
    mask255 = (np.isin(seg, building_ids)).astype(np.uint8)*255

    # 2) 抠图 + 去阴影
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[mask255==0, 3] = 0
    sh = shadow_mask_lab(img_bgr, mask255)
    bgra[sh==255, 3] = 0

    # 3) 提色 + 右侧拼接
    colors = get_palette_from_bgra(bgra)
    out = compose_with_palette_keep_alpha(bgra, colors)  # BGRA

    # 返回 base64 PNG + 颜色 JSON
    ok, buf = cv2.imencode(".png", out)
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    data_url = "data:image/png;base64," + b64
    return data_url, colors

with gr.Blocks() as demo:
    gr.Markdown("## Urban Facade Color (CPU Space)\n上传街景图获得建筑立面主色（右侧色卡）")
    inp = gr.Image(type="pil", label="Upload image")
    btn = gr.Button("Analyze")
    out_img = gr.Image(label="Result w/ palette (right)")
    out_json = gr.JSON(label="Palette [RGB, ratio]")
    btn.click(pipeline, inputs=inp, outputs=[out_img, out_json])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
