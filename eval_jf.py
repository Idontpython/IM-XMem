import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt as edt
import math

def compute_iou(gt_mask, pred_mask):
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return inter / union if union > 0 else 1.0  # 空对象时定义 IoU=1

def compute_boundary_f(gt_mask, pred_mask, tol_px):
    # 提取二值轮廓
    k = np.ones((3,3), np.uint8)
    gt_erode = cv2.erode(gt_mask, k, 1); pred_erode = cv2.erode(pred_mask, k, 1)
    b_gt = (gt_mask ^ gt_erode).astype(np.uint8)
    b_pd = (pred_mask ^ pred_erode).astype(np.uint8)

    # 计算距离变换
    dt_gt, dt_pd = edt(1-b_gt), edt(1-b_pd)

    # 统计匹配的边界像素
    match_gt2pd = (b_gt * (dt_pd <= tol_px)).sum()
    match_pd2gt = (b_pd * (dt_gt <= tol_px)).sum()
    P = match_pd2gt / max(1, b_pd.sum()); R = match_gt2pd / max(1, b_gt.sum())

    return 0.0 if (P+R)==0 else 2*P*R/(P+R)

def evaluate(gt_dir, pred_dir):
    gt_files   = sorted([f for f in os.listdir(gt_dir)   if f.endswith('.png')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    assert gt_files == pred_files, "GT 和 pred 文件不匹配"

    ious, fs = [], []

    # 以首帧 GT 的对象集合为评估对象集合（与 DAVIS 半监督评估口径一致）
    if len(gt_files) == 0:
        print('No frames found.')
        return
    first_gt = np.array(Image.open(os.path.join(gt_dir, gt_files[0])).convert('P'))
    eval_labels = [lid for lid in np.unique(first_gt) if lid != 0]

    print("Frame\tIoU\tF")
    for fn in tqdm(gt_files):
        gt_idx = np.array(Image.open(os.path.join(gt_dir, fn)).convert('P'))  # 索引掩码（背景=0，其它为对象id）
        pd_idx = np.array(Image.open(os.path.join(pred_dir, fn)).convert('P'))

        H, W = gt_idx.shape
        # DAVIS 边界容差：0.008 * 图像对角线
        tol_px = max(1, int(round(0.008 * math.hypot(H, W))))

        # 逐对象计算并取平均（DAVIS 按对象平均口径）
        per_obj_js, per_obj_fs = [], []
        for lid in eval_labels:
            g = (gt_idx == lid).astype(np.uint8)
            p = (pd_idx == lid).astype(np.uint8)

            # J: IoU（若对象同时在GT与Pred均为空，记为1.0；若仅一侧为空，记为0.0）
            inter = np.logical_and(g==1, p==1).sum()
            union = np.logical_or(g==1, p==1).sum()
            if g.sum()==0 and p.sum()==0:
                j = 1.0
            elif union == 0:
                j = 0.0
            else:
                j = inter/union
            per_obj_js.append(j)

            # F: 边界 F，使用距离变换匹配 + 0.008×对角线的容差
            if g.sum()==0 and p.sum()==0:
                f_obj = 1.0
            elif g.sum()==0 or p.sum()==0:
                f_obj = 0.0
            else:
                f_obj = compute_boundary_f(g, p, tol_px)
            per_obj_fs.append(f_obj)

        iou = float(np.mean(per_obj_js))
        f   = float(np.mean(per_obj_fs))

        ious.append(iou)
        fs.append(f)
        print(f"{fn}\t{iou:.4f}\t{f:.4f}")

    mean_j = np.mean(ious)
    mean_f = np.mean(fs)
    print("\n===== Overall =====")
    print(f"Mean J (IoU): {mean_j:.4f}")
    print(f"Mean F (boundary): {mean_f:.4f}")
    print(f"J & F mean: {(mean_j+mean_f)/2:.4f}")

if __name__ == "__main__":
    # TODO: 根据你的路径修改下面两行
    gt_dir   = ""
    pred_dir = ""
    evaluate(gt_dir, pred_dir) 
