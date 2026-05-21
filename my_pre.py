import numpy as np
import cv2
import os
import natsort
from pathlib import Path

# ==============================================================================
# 配置路径 - 请确保你的图片放在 ROOT_DIR/任意文件夹/images 目录下
# ==============================================================================
ROOT_DIR = Path("/root/autodl-tmp/bi/DGUNet")

# ==============================================================================
# 性能优化版算法函数
# ==============================================================================

def getMaxChannel(img, blockSize):
    # 【性能优化】使用 OpenCV 膨胀操作替代原始的三重循环，速度提升万倍
    kernel = np.ones((blockSize, blockSize), np.uint8)
    return cv2.dilate(img, kernel)

def getMAxChannel(img):
    # 【性能优化】直接使用 numpy 的 max 轴操作
    return np.max(img[:, :, :2], axis=2)

class GuidedFilter:
    def __init__(self, I, radius, epsilon):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = self._toFloatImg(I)
        self._initFilter()

    def _toFloatImg(self, img):
        if img.dtype == np.float32: return img
        return (1.0 / 255.0) * np.float32(img)

    def _initFilter(self):
        I, r, eps = self._I, self._radius, self._epsilon
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]
        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))
        Irr_var = cv2.blur(Ir**2, (r, r)) - self._Ir_mean**2 + eps
        Irg_var = cv2.blur(Ir*Ig, (r, r)) - self._Ir_mean*self._Ig_mean
        Irb_var = cv2.blur(Ir*Ib, (r, r)) - self._Ir_mean*self._Ib_mean
        Igg_var = cv2.blur(Ig**2, (r, r)) - self._Ig_mean**2 + eps
        Igb_var = cv2.blur(Ig*Ib, (r, r)) - self._Ig_mean*self._Ib_mean
        Ibb_var = cv2.blur(Ib**2, (r, r)) - self._Ib_mean**2 + eps
        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var
        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
        self._Irr_inv, self._Irg_inv, self._Irb_inv = Irr_inv/I_cov, Irg_inv/I_cov, Irb_inv/I_cov
        self._Igg_inv, self._Igb_inv, self._Ibb_inv = Igg_inv/I_cov, Igb_inv/I_cov, Ibb_inv/I_cov

    def _computeCoefficients(self, p):
        r, I = self._radius, self._I
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]
        p_mean = cv2.blur(p, (r, r))
        Ipr_mean, Ipg_mean, Ipb_mean = cv2.blur(Ir*p, (r, r)), cv2.blur(Ig*p, (r, r)), cv2.blur(Ib*p, (r, r))
        Ipr_cov, Ipg_cov, Ipb_cov = Ipr_mean - self._Ir_mean*p_mean, Ipg_mean - self._Ig_mean*p_mean, Ipb_mean - self._Ib_mean*p_mean
        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov
        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean
        return cv2.blur(ar, (r, r)), cv2.blur(ag, (r, r)), cv2.blur(ab, (r, r)), cv2.blur(b, (r, r))

    def filter(self, p):
        ar_m, ag_m, ab_m, b_m = self._computeCoefficients(p)
        return ar_m * self._I[:,:,0] + ag_m * self._I[:,:,1] + ab_m * self._I[:,:,2] + b_m

# ==============================================================================
# 物理建模函数
# ==============================================================================

def get_attenuation(image, gamma=1.2):
    att = 1 - image**(gamma)
    mean_att = np.mean(att, axis=(0, 1))
    return np.argsort(mean_att) # B, G, R 顺序

def DepthMap(img, blockSize, index_att):
    img_c_star = img[:, :, index_att[-1]]
    img_c_other = getMAxChannel(img)
    max_c_star = getMaxChannel(img_c_star, blockSize)
    max_c = getMaxChannel(img_c_other, blockSize)
    return max_c_star - max_c

def Refinedtransmission(transmission, img):
    gfilt = GuidedFilter(img, 50, 0.001)
    return np.clip(gfilt.filter(transmission), 0, 255)

def estimateBackgroundLight(image):
    index = get_attenuation(image)
    depth = DepthMap(image, 9, index)
    # 取深度图中值最小的像素位置作为背景光估计点
    min_idx = np.unravel_index(np.argmin(depth, axis=None), depth.shape)
    return image[min_idx[0], min_idx[1], :]

# ==============================================================================
# 主循环
# ==============================================================================

if __name__ == '__main__':
    # 【改这里】直接指向你存放图片的文件夹路径
    img_input_path = "/root/autodl-tmp/bi/ROUD/light" 
    
    # 【改这里】定义结果保存到哪里
    save_base_dir = "/root/autodl-tmp/bi/DGUNet/pre_light"

    if not os.path.exists(img_input_path):
        print(f"❌ 找不到图片文件夹: {img_input_path}")
        exit()

    # 获取图片列表
    files = [f for f in os.listdir(img_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files = natsort.natsorted(files)

    # 创建输出目录
    t_prior_dir = os.path.join(save_base_dir, 't_prior')
    B_prior_dir = os.path.join(save_base_dir, 'B_prior')
    os.makedirs(t_prior_dir, exist_ok=True)
    os.makedirs(B_prior_dir, exist_ok=True)

    print(f"找到 {len(files)} 张图片，准备处理...")

    for file in files:
        filepath = os.path.join(img_input_path, file)
        prefix = os.path.splitext(file)[0]

        img = cv2.imread(filepath)
        if img is None: continue

        # --- 以下是核心处理逻辑，保持不变 ---
        image = img / 255.0
        index = get_attenuation(image)
        largestDiff = DepthMap(image, 7, index)
        transmission = largestDiff + (1 - np.max(largestDiff))
        transmission = np.clip(transmission, 0.1, 0.9) 
        transmission = Refinedtransmission(transmission * 255, image * 255)
        bg_light = estimateBackgroundLight(image)
        B_img = np.ones(image.shape) * bg_light * 255

        # 保存结果
        cv2.imwrite(os.path.join(t_prior_dir, f"{prefix}.jpg"), np.uint8(transmission))
        cv2.imwrite(os.path.join(B_prior_dir, f"{prefix}.jpg"), np.uint8(B_img))
        print(f"已生成: {file} 的先验图")

    print("\n✅ 处理完成！")
    print(f"请在测试配置文件中将路径指向: {save_base_dir}")