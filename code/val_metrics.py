# /home/xushutan/XMem/code/val_metrics.py
import os, glob, math, argparse, json
from pathlib import Path
import numpy as np
import torch
torch.set_grad_enabled(False)
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from inference.data.mask_mapper import MaskMapper

from model.network import XMem
from inference.inference_core import InferenceCore
from dataset.range_transform import im_normalization

def load_frames(img_dir):
    frames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
    return frames

def read_image(path):
    im = Image.open(path).convert('RGB')
    im = torch.from_numpy(np.array(im)).float().permute(2,0,1)/255.0
    im = im_normalization(im)
    return im

def read_mask(path):
    # PNG 标签：0 背景，其它为对象 id
    m = np.array(Image.open(path).convert('P'), dtype=np.uint8)
    return m

def make_onehot_mask(gt, labels):
    # labels: 连续对象 id 列表（不含0），输出 num_obj x H x W 的 {0,1} mask
    ms = []
    for lid in labels:
        ms.append((gt==lid).astype(np.float32))
    if len(ms)==0:
        return None
    m = np.stack(ms, 0)
    return torch.from_numpy(m)

def ce_from_prob_with_bg(prob_with_bg, gt):
    # prob_with_bg: CxHxW，包含背景；gt: HxW uint8，值域 0..C-1
    eps = 1e-6
    C,H,W = prob_with_bg.shape
    gt_t = torch.from_numpy(gt.astype(np.int64)).to(prob_with_bg.device)
    p = torch.clamp(prob_with_bg.permute(1,2,0), eps, 1.0)  # HxWxC
    ll = -torch.log(p[torch.arange(H)[:,None], torch.arange(W)[None,:], gt_t])  # HxW
    return ll.mean().item()

def iou_score(pred, gt, labels):
    # 多对象平均 J
    ious = []
    for lid in labels:
        p = (pred==lid)
        g = (gt==lid)
        inter = (p & g).sum()
        union = (p | g).sum()
        if p.sum()==0 and g.sum()==0:
            ious.append(1.0)
        else:
            ious.append(0.0 if union==0 else inter/union)
    return float(np.mean(ious)) if ious else 0.0

def boundary_map(bw):
    bw = bw.astype(np.uint8)
    # 形态学梯度近似轮廓
    kernel = np.ones((3,3),np.uint8)
    dil = cv2.dilate(bw, kernel, iterations=1)
    ero = cv2.erode(bw, kernel, iterations=1)
    return (dil!=ero).astype(np.uint8)

def f_boundary(pred, gt, labels, tol_ratio=0.008):
    # 多对象平均 F，带容差
    H,W = gt.shape
    tol = max(1, int(round(tol_ratio*math.sqrt(H*H+W*W))))
    Fs = []
    for lid in labels:
        pb = boundary_map((pred==lid))
        gb = boundary_map((gt==lid))
        if pb.sum()==0 and gb.sum()==0:
            Fs.append(1.0); continue
        if pb.sum()==0 or gb.sum()==0:
            Fs.append(0.0); continue
        # 容差膨胀
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
        gb_d = cv2.dilate(gb, k, iterations=1)
        pb_d = cv2.dilate(pb, k, iterations=1)
        tp_p = (pb & gb_d).sum()
        tp_g = (gb & pb_d).sum()
        P = tp_p / max(1, pb.sum())
        R = tp_g / max(1, gb.sum())
        Fm = 0.0 if (P+R)==0 else 2*P*R/(P+R)
        Fs.append(Fm)
    return float(np.mean(Fs)) if Fs else 0.0

def boundary_f_official(pred, gt, labels):
    import cv2, math
    from scipy.ndimage import distance_transform_edt as edt

    def bin_boundary(bw):
        k = np.ones((3,3), np.uint8)
        e = cv2.erode(bw.astype(np.uint8), k, 1)
        return (bw.astype(np.uint8) ^ e).astype(np.uint8)

    H, W = gt.shape
    # DAVIS 边界容差：0.008 * 图像对角线
    tol_px = max(1, int(round(0.008 * math.hypot(H, W))))

    Fs = []
    for lid in labels:
        p = (pred==lid).astype(np.uint8)
        g = (gt==lid).astype(np.uint8)
        if p.sum()==0 and g.sum()==0: Fs.append(1.0); continue
        if p.sum()==0 or g.sum()==0: Fs.append(0.0); continue

        pb, gb = bin_boundary(p), bin_boundary(g)
        dt_g, dt_p = edt(1-gb), edt(1-pb)
        match_p = (pb * (dt_g <= tol_px)).sum()
        match_g = (gb * (dt_p <= tol_px)).sum()
        P = match_p / max(1, pb.sum())
        R = match_g / max(1, gb.sum())
        Fs.append(0.0 if (P+R)==0 else 2*P*R/(P+R))
    return float(np.mean(Fs)) if Fs else 0.0

def evaluate_checkpoint(model_path, data_root, device='cuda', size=480):
    cfg = {
        'mem_every': 10, 'deep_update_every': -1,
        'enable_long_term': False,  # 评估时可关掉以省显存
        'enable_long_term_count_usage': False,
        'max_mid_term_frames': 10, 'min_mid_term_frames': 5,
        'max_long_term_elements': 10000, 'num_prototypes': 128,
        'top_k': 30, 'hidden_dim': 64,
    }
    net = XMem(cfg, model_path).to(device).eval()
    processor = InferenceCore(net, config=cfg)

    with torch.no_grad(), torch.cuda.amp.autocast(True):
        img_root = Path(data_root)/'JPEGImages'
        ann_root = Path(data_root)/'Annotations'
        videos = sorted([v for v in os.listdir(img_root) if (img_root/v).is_dir() and (ann_root/v).is_dir()])

        total_ce, total_frames = 0.0, 0
        Js, Fs = [], []

        for vid in videos:
            frames = load_frames(img_root/vid)
            if not frames: continue
            processor.clear_memory()
            first_done = False
            mapper = MaskMapper()  # 映射固定为“首帧对象集合”

            for ti, fname in enumerate(frames):
                rgb = read_image(img_root/vid/fname).to(device)
                if size > 0:
                    C,H,W = rgb.shape
                    short = min(H,W)
                    if short != size:
                        scale = size/short
                        rgb = F.interpolate(rgb.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False)[0]

                gt = read_mask((ann_root/vid/(fname[:-4]+'.png')))

                if not first_done:
                    # 用首帧建立映射与 one-hot，半监督只评估首帧对象
                    msk_t, labels_mapped = mapper.convert_mask(gt)  # one-hot 形状: num_obj x H x W（不含背景）
                    processor.set_all_labels(list(mapper.remappings.values()))
                    prob = processor.step(rgb, msk_t.to(device), labels_mapped, end=(ti==len(frames)-1))
                    first_done = True
                else:
                    prob = processor.step(rgb, None, None, end=(ti==len(frames)-1))

                if prob is None:
                    continue

                # 将 GT 映射到连续索引，背景=0，其余=1..K
                mapped_gt = np.zeros_like(gt, dtype=np.uint8)
                for orig, mapped in mapper.remappings.items():
                    mapped_gt[gt == orig] = mapped

                # 全像素 CE（与网络通道对齐：C=K+1，含背景）
                total_ce += ce_from_prob_with_bg(prob, mapped_gt)
                total_frames += 1

                # 预测索引与 J/F（仅对首帧对象集合评估）
                pred = torch.argmax(prob, dim=0).detach().cpu().numpy().astype(np.uint8)
                eval_labels = list(mapper.remappings.values())
                Js.append(iou_score(pred, mapped_gt, eval_labels))
                Fs.append(boundary_f_official(pred, mapped_gt, eval_labels))

    avg_ce = total_ce / max(1, total_frames)
    jf = float(np.mean([(j+f)/2 for j,f in zip(Js,Fs)])) if Js else 0.0
    # 释放显存
    del processor, net
    torch.cuda.empty_cache()
    return avg_ce, float(np.mean(Js)) if Js else 0.0, float(np.mean(Fs)) if Fs else 0.0, jf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models_glob', required=True)
    ap.add_argument('--val_root', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--size', type=int, default=480)   # -1 保持原分辨率
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # 原来
    ckpts = sorted(glob.glob(args.models_glob))
    # 改为：排除 checkpoint 文件
    ckpts = [p for p in sorted(glob.glob(args.models_glob)) if 'checkpoint' not in os.path.basename(p)]
    steps = []
    rows = []
    for mp in ckpts:
        step = Path(mp).stem.split('_')[-1]  # 1000,2000,...
        print(f'Evaluating {mp} (step={step})')
        ce,j,f,jf = evaluate_checkpoint(mp, args.val_root, device=args.device, size=args.size)
        steps.append(int(step)); rows.append((int(step), ce, j, f, jf))

    rows.sort(key=lambda x:x[0])

    # 写CSV
    csv_path = os.path.join(args.out_dir, 'metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('step,ce,j,f,jf\n')
        for r in rows:
            f.write(','.join([str(x) for x in r])+'\n')
    print('Saved', csv_path)

    # 画图
    xs = [r[0] for r in rows]
    ce = [r[1] for r in rows]
    jf = [r[4] for r in rows]

    plt.figure(); plt.plot(xs, ce, marker='o'); plt.xlabel('checkpoint step'); plt.ylabel('CE (all pixels)'); plt.grid(True)
    p1 = os.path.join(args.out_dir, 'loss_vs_ckpt.png'); plt.savefig(p1, dpi=150); print('Saved', p1)

    plt.figure(); plt.plot(xs, jf, marker='o'); plt.xlabel('checkpoint step'); plt.ylabel('J&F'); plt.grid(True)
    p2 = os.path.join(args.out_dir, 'jf_vs_ckpt.png'); plt.savefig(p2, dpi=150); print('Saved', p2)

if __name__ == '__main__':
    main()