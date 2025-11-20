# segment_building.py
import sys, glob, csv
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans  # 新增：提色

from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules
import torch

# ===== 路径配置 =====
CFG_PATH  = 'segformer_mit-b0_8xb2-160k_ade20k-512x512.py'   # 已下载在根目录
CKPT_GLOB = 'segformer_mit-b0_*ade20k*.pth'                  # 自动匹配权重
IN_DIR    = 'images'                                         # 原图
OUT_MASK  = 'images_result'                                  # Step1：通道图（建筑=255 其它=0）
OUT_ONLY  = 'images_overlay'                                 # Step2&3：去阴影后的透明PNG
OUT_PALETTE_DIR = 'pic_color'                                # Step4：拼接色卡后的成品
CSV_OUT  = 'color_summary.csv'
# ====================

# ===== 阴影检测参数（可按需要微调）=====
KL = 1.0   # L 通道 z 分数阈值：L < mean(L) - KL*std(L)
KB = 0.5   # B 通道 z 分数阈值：B < mean(B) - KB*std(B)
MORPH_KERNEL = 3  # 形态学开运算 kernel 尺寸（像素），设 0 关闭
# =====================================

# ===== 提色与色卡参数 =====
TOPK = 5            # 主色个数
PALETTE_W = 120     # 色卡宽度(px)
WHITE_TH = 240      # 过滤近白
BLACK_TH = 20       # 过滤近黑
MIN_SAMPLES = 500   # 建筑有效像素下限
# ========================

def find_ckpt():
    cands = sorted(glob.glob(CKPT_GLOB))
    if not cands:
        print('❗未找到权重文件，请先执行：\n'
              'mim download mmsegmentation --config segformer_mit-b0_8xb2-160k_ade20k-512x512 --dest .')
        sys.exit(1)
    return cands[0]

def pick_building_ids(classes):
    STRICT = {'building','house','skyscraper','garage','roof','windowpane','door','balcony'}
    EXCLUDE = {'fence','railing','wall','arch','column','beam'}  # 可按需要留/删
    name2id = {n:i for i,n in enumerate(classes)}
    ids = [i for n,i in name2id.items() if n in STRICT]
    if 'building' in name2id and name2id['building'] not in ids:
        ids.append(name2id['building'])
    return sorted(set(ids))

def ensure_dirs(*dirs):
    for d in dirs: Path(d).mkdir(parents=True, exist_ok=True)

def shadow_mask_lab(img_bgr, valid_mask255):
    """CIELAB 阴影检测：L<mean- KL*std 且 B<mean- KB*std（仅在建筑区域内统计与检测）"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, _, B = lab[..., 0], lab[..., 1], lab[..., 2]
    m = valid_mask255 == 255
    if not np.any(m):
        return np.zeros_like(L, dtype=np.uint8)
    Lm, Bm = L[m], B[m]
    L_mean, L_std = float(Lm.mean()), float(Lm.std() + 1e-6)
    B_mean, B_std = float(Bm.mean()), float(Bm.std() + 1e-6)
    shadow = ((L < (L_mean - KL * L_std)) & (B < (B_mean - KB * B_std)) & m).astype(np.uint8) * 255
    if MORPH_KERNEL and MORPH_KERNEL > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, k, iterations=1)
    return shadow

def save_building_only_shadowfree(img_bgr, mask255, out_path):
    """建筑抠图 + 阴影透明"""
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[mask255 == 0, 3] = 0
    sh_mask = shadow_mask_lab(img_bgr, mask255)
    bgra[sh_mask == 255, 3] = 0
    cv2.imwrite(str(out_path), bgra)

# ---------- Step4: 提色并拼接色卡 ----------
def load_rgba(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        a = np.full(img.shape[:2], 255, np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[..., 3] = a
    return img[..., :3], img[..., 3]   # BGR, A

def get_dominant_colors(bgr, alpha, k=TOPK):
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
    n_clusters = int(min(k, max(1, len(uniq))))
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    km.fit(sel.astype(np.float32))
    centers = km.cluster_centers_.clip(0, 255).astype(np.uint8)
    counts = np.bincount(km.labels_, minlength=n_clusters).astype(np.float64)
    ratios = counts / counts.sum()
    order = np.argsort(-ratios)
    return [(centers[i].tolist(), float(ratios[i])) for i in order]

def compose_with_palette_keep_alpha(bgra, colors, palette_w=PALETTE_W):
    """保留左侧透明通道，右侧色卡为不透明；输出 BGRA"""
    h, w = bgra.shape[:2]
    # 色卡用不透明BGRA
    card = np.zeros((h, palette_w, 4), np.uint8)
    card[..., 3] = 255  # 右侧色卡全不透明
    if colors:
        y = 0
        for rgb, ratio in colors:
            bh = max(1, int(round(ratio * h)))
            bgr = (rgb[2], rgb[1], rgb[0], 255)
            card[y:y+bh, :] = bgr
            y += bh
        if y < h:
            card[y:h, :] = card[y-1, :] if y > 0 else (60, 60, 60, 255)
    # 横向拼接（左：BGRA；右：BGRA）
    out = np.concatenate([bgra, card], axis=1)
    return out
# -----------------------------------------

def main():
    # 1) 读取原图列表
    in_dir = Path(IN_DIR)
    imgs = [p for p in in_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]
    if not imgs:
        print(f'❗{IN_DIR} 里没有图片'); sys.exit(1)

    ensure_dirs(OUT_MASK, OUT_ONLY, OUT_PALETTE_DIR)

    # 2) 若无通道图则分割推理
    need_infer = any(not (Path(OUT_MASK) / f'{p.stem}_building.png').exists() for p in imgs)
    if need_infer:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        register_all_modules()
        ckpt = find_ckpt()
        model = init_model(CFG_PATH, ckpt, device=device)
        classes = model.dataset_meta.get('classes')
        building_ids = [1] if classes is None else pick_building_ids(classes)
        print(f'使用设备: {device} | 建筑相关ID: {building_ids}')
        for p in tqdm(imgs, desc='Step1: Segment -> building channel'):
            img_bgr = cv2.imread(str(p));  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = inference_model(model, img_rgb)
            seg = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int32)
            mask255 = (np.isin(seg, building_ids)).astype(np.uint8) * 255
            cv2.imwrite(str(Path(OUT_MASK) / f'{p.stem}_building.png'), mask255)
        print(f'✅ 通道图已保存到：{OUT_MASK}')
    else:
        print('ℹ️ 检测到现有通道图，跳过分割推理。')

    # 3) 抠建筑并去阴影
    for p in tqdm(imgs, desc='Step2: Keep building only & remove shadows'):
        img_bgr = cv2.imread(str(p))
        mask_path = Path(OUT_MASK) / f'{p.stem}_building.png'
        if img_bgr is None or not mask_path.exists():
            continue
        mask255 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        out_path = Path(OUT_ONLY) / f'{p.stem}_building_shadowfree.png'
        save_building_only_shadowfree(img_bgr, mask255, out_path)
    print(f'✅ 去阴影的建筑透明图已保存到：{OUT_ONLY}')

    # 4) 提取主色并拼接色卡到右侧
    csv_path = Path(OUT_PALETTE_DIR) / CSV_OUT
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv); writer.writerow(["file", "palette_rgb", "ratios"])
        files = sorted(Path(OUT_ONLY).glob("*.png"))  # 改动 1
        for fp in tqdm(files, desc='Step3: Palette & compose'):
            bgr, alpha = load_rgba(fp)
            if bgr is None:
                continue
            colors = get_dominant_colors(bgr, alpha, k=TOPK)

            # 还原 BGRA（左侧保持透明区域）
            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            bgra[alpha == 0, 3] = 0

            # 透明拼接
            out_img = compose_with_palette_keep_alpha(bgra, colors, PALETTE_W)

            # 保存 PNG 到 pic_color
            out_path = Path(OUT_PALETTE_DIR) / f"{fp.stem.replace('_building_shadowfree','')}_palette.png"
            cv2.imwrite(str(out_path), out_img)

            writer.writerow([fp.name, [c for c,_ in colors], [r for _,r in colors]])

    print(f'✅ 色卡成品已保存到：{OUT_PALETTE_DIR}')
    print(f'📝 颜色统计CSV：{csv_path}')

if __name__ == '__main__':
    main()
