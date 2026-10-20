import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from pytorch_msssim import ssim

# 引入你的自定义配置模块
from Config.vgg import *
from Config.options_1 import opt, device
from Config.models_1 import *
from Config.misc import *

# ==========================================
# 1. 定义必要的损失函数与工具函数
# ==========================================
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        sobel_x = torch.tensor([[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]]], dtype=torch.float32)
        self.sobel_x = sobel_x.unsqueeze(0)
        self.sobel_y = sobel_y.unsqueeze(0)
        self.register_buffer('weight_x', self.sobel_x)
        self.register_buffer('weight_y', self.sobel_y)

    def forward(self, pred, target):
        def to_gray(x):
            return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

        pred_gray = to_gray(pred)
        target_gray = to_gray(target)

        grad_pred_x = F.conv2d(pred_gray, self.weight_x, padding=1)
        grad_pred_y = F.conv2d(pred_gray, self.weight_y, padding=1)
        grad_target_x = F.conv2d(target_gray, self.weight_x, padding=1)
        grad_target_y = F.conv2d(target_gray, self.weight_y, padding=1)

        grad_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2 + 1e-6)
        grad_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2 + 1e-6)

        return F.l1_loss(grad_pred, grad_target)

def color_stats_loss(pred, gt):
    pred_mean = pred.mean(dim=[2, 3])
    gt_mean = gt.mean(dim=[2, 3])
    mean_loss = ((pred_mean - gt_mean) ** 2).mean()

    pred_std = pred.std(dim=[2, 3])
    gt_std = gt.std(dim=[2, 3])
    std_loss = ((pred_std - gt_std) ** 2).mean()

    return mean_loss + std_loss

def remove_module_prefix(state_dict):
    """删除 DataParallel 带来的 'module.' 前缀"""
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def compute_dolp_aolp(I0, I45, I90, eps=1e-6, mask_threshold=0.1):
    gray = lambda x: 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    I0_gray, I45_gray, I90_gray = gray(I0), gray(I45), gray(I90)
    S0 = I0_gray + I90_gray
    S1 = I0_gray - I90_gray
    S2 = 2 * I45_gray - I0_gray - I90_gray
    DoLP = torch.sqrt(S1**2 + S2**2 + eps) / (S0 + eps)
    AoLP = 0.5 * torch.atan2(S2, S1)
    AoLP = (AoLP + math.pi / 2) / math.pi
    AoLP_sin = torch.sin(2 * math.pi * AoLP)
    AoLP_cos = torch.cos(2 * math.pi * AoLP)
    mask = (S0 > mask_threshold).float()
    return DoLP * mask, AoLP_sin * mask, AoLP_cos * mask, mask

# ==========================================
# 2. 数据集定义 (仅保留验证集所需逻辑)
# ==========================================
class ProcessedS0Dataset(Dataset):
    def __init__(self, root_dir, mode='val'):
        self.root_dir = os.path.join(root_dir, 'processed_images', mode)
        self.files = os.listdir(self.root_dir)
        self.sample_keys = []
        self.mode = mode
        
        for f in self.files:
            if '_I0_' in f:
                parts = f.rsplit('_', 2)
                target = parts[0]
                turbidity = parts[2].replace('.png', '')
                self.sample_keys.append((target, turbidity))
                
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        target, turbidity = self.sample_keys[idx]

        def load_img(direction):
            fname = f"{target}_{direction}_{turbidity}.png"
            fpath = os.path.join(self.root_dir, fname)
            return self.transform(Image.open(fpath).convert('RGB'))

        return {
            'I0': load_img('I0'), 'I45': load_img('I45'), 'I90': load_img('I90'),
            'GT0': load_img('GT0'), 'GT45': load_img('GT45'), 'GT90': load_img('GT90'),
            'target': target,
            'turbidity': turbidity
        }

# ==========================================
# 3. 验证主程序
# ==========================================
if __name__ == '__main__':
    # --- 替换为你实际的权重路径 ---
    WEIGHTS_PATH = os.path.join(opt.best_dir, 'DICAM_best_epoch105_loss0.454433.pth')
    
    
    SAVE_DIR = 'test_results'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 初始化模型与损失函数
    print("🚀 Initializing model for validation...")
    model1 = DICAM().to(device)
    vgg = Vgg19(requires_grad=False).to(device)
    mae_loss = nn.L1Loss()
    edge_loss_fn = EdgeLoss().to(device)

    # 2. 加载权重文件
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"❌ Weights file not found: {WEIGHTS_PATH}")
    
    print(f"📦 Loading weights from: {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    raw_state_dict = checkpoint['model_state_dict']
    
    # 自动处理可能存在的 module. 前缀
    if any(k.startswith('module.') for k in raw_state_dict):
        raw_state_dict = remove_module_prefix(raw_state_dict)
        
    model1.load_state_dict(raw_state_dict, strict=True)
    print("✅ Model weights loaded successfully!")

    # 3. 准备数据
    val_set = ProcessedS0Dataset(root_dir=opt.root_dir, mode='val')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    print(f"📊 Found {len(val_set)} images in validation set.")

    # 4. 开始验证
    model1.eval()
    
    # 初始化统计变量
    val_psnr_total, val_ssim_total = 0.0, 0.0
    val_total_mae_loss, val_total_color_loss = 0.0, 0.0
    val_total_ssim_loss, val_total_vgg_loss, val_total_edge_loss = 0.0, 0.0, 0.0
    val_count = 0

    with torch.no_grad():
        for i, val_batch in enumerate(tqdm(val_loader, desc='Testing', ncols=100)):
            # 加载数据到设备
            I0 = val_batch['I0'].to(device)
            I45 = val_batch['I45'].to(device)
            I90 = val_batch['I90'].to(device)
            GT0 = val_batch['GT0'].to(device)
            GT90 = val_batch['GT90'].to(device)
            
            target = val_batch['target'][0]
            turbidity = val_batch['turbidity'][0]

            # 计算偏振特征与真值
            DoLp, AoLP_1, AoLP_2, mask = compute_dolp_aolp(I0, I45, I90)
            S0 = ((I0 + I90) / 2).clamp(0.0, 1.0) * mask
            GT_S0 = ((GT0 + GT90) / 2).clamp(0.0, 1.0) * mask

            # 网络前向传播
            pred = model1(DoLp, AoLP_1, AoLP_2, S0)
            pred = pred * mask

            # 计算各项 Loss 
            val_mae = mae_loss(pred, GT_S0)
            val_color = color_stats_loss(pred, GT_S0)
            raw_ssim_val = torch.clamp(ssim(pred, GT_S0, data_range=1, size_average=True), 0.0, 1.0)
            ssim_loss_val = 1 - raw_ssim_val
            
            vgg_gt = vgg(normalize_batch(GT_S0))
            vgg_pred = vgg(normalize_batch(pred))
            vgg_loss_val = mae_loss(vgg_pred.relu4_3, vgg_gt.relu4_3)
            edge_loss_val = edge_loss_fn(pred, GT_S0)

            # 累加 Loss
            val_total_mae_loss += val_mae.item()
            val_total_color_loss += val_color.item()
            val_total_ssim_loss += ssim_loss_val.item()
            val_total_vgg_loss += vgg_loss_val.item()
            val_total_edge_loss += edge_loss_val.item()

            # 图像张量转 Numpy 计算 PSNR/SSIM 指标
            def to_np(t): 
                return np.clip(t.squeeze(0).cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)

            pred_np = to_np(pred)
            gt_np = to_np(GT_S0)

            psnr_val = compare_psnr(gt_np, pred_np, data_range=1.0)
            ssim_val = compare_ssim(gt_np, pred_np, multichannel=True, data_range=1.0)

            val_psnr_total += psnr_val
            val_ssim_total += ssim_val
            val_count += 1

            # 格式化保存输出图像
            psnr_str = f"{psnr_val:.2f}"
            ssim_str = f"{ssim_val:.4f}"
            filename = f"{target}_{turbidity}_psnr{psnr_str}_ssim{ssim_str}.png"
            path = os.path.join(SAVE_DIR, filename)
            
            torchvision.utils.save_image(pred.squeeze(0).cpu(), path)

    # 5. 输出最终统计结果
    mean_mae = val_total_mae_loss / val_count
    mean_ssim = val_total_ssim_loss / val_count
    mean_vgg = val_total_vgg_loss / val_count
    mean_edge = val_total_edge_loss / val_count
    mean_color = val_total_color_loss / val_count

    # 注意：这里的权重系数依赖于你的 opt 配置文件，如果不需要总 Loss，这部分可以注释掉
    mean_total_loss = (
        opt.lambda_mae * mean_mae +
        opt.lambda_vgg * mean_vgg +
        opt.lambda_edge * mean_edge +
        opt.lambda_color * mean_color +
        opt.lambda_ssim * mean_ssim 
    )

    print("\n" + "="*50)
    print("🏆 Validation Finished!")
    print(f"Results saved to: {SAVE_DIR}/")
    print(f"Total Evaluated Images: {val_count}")
    print(f"Final PSNR: {val_psnr_total / val_count:.2f} dB")
    print(f"Final SSIM: {val_ssim_total / val_count:.4f}")
    print("="*50)
    print(f"Loss Details:")
    print(f"MAE: {mean_mae:.6f} | SSIM_Loss: {mean_ssim:.6f} | VGG: {mean_vgg:.6f}")
    print(f"EDGE: {mean_edge:.6f} | COLOR: {mean_color:.6f} | Weighted_Total: {mean_total_loss:.6f}")