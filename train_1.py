import argparse
import cv2
import os
from torch.utils.data import Sampler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image
from Config.vgg import *
from Config.options_1 import opt, device
from Config.models_1 import *
from Config.misc import *
from torchsummary import summary
from numba import cuda
from pytorch_msssim import ssim
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import math
from torch.utils.data import WeightedRandomSampler


def get_lr(optimizer):
	# 取出学习率
	for param_group in optimizer.param_groups:
		return param_group['lr']
    
    
class DynamicWeightedSampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def set_weights(self, new_weights):
        self.weights = torch.tensor(new_weights, dtype=torch.double)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

    
def color_stats_loss(pred, gt):
    """
    颜色一致性损失（有真值参与）：
    通道均值差 + 通道标准差差
    """
    pred_mean = pred.mean(dim=[2, 3])
    gt_mean = gt.mean(dim=[2, 3])
    mean_loss = ((pred_mean - gt_mean) ** 2).mean()

    pred_std = pred.std(dim=[2, 3])
    gt_std = gt.std(dim=[2, 3])
    std_loss = ((pred_std - gt_std) ** 2).mean()

    return mean_loss + std_loss
    
    
def get_dynamic_turbidity_weights(epoch, max_epoch=150):
    base_weights = {'1.00': 1, '1.76': 2, '2.52': 3, '3.28': 4, '4.04': 5}
    scale = min(epoch / (0.4 * max_epoch), 1.0)  # 前 40% epoch 逐渐放大高浊度权重
    dynamic_weights = {}
    for k, v in base_weights.items():
        dynamic_weights[k] = 1 + scale * (v - 1)  # 逐渐从 1→v 过渡
    return dynamic_weights



class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        sobel_x = torch.tensor([[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]]], dtype=torch.float32)
        self.sobel_x = sobel_x.unsqueeze(0)  # shape: [1,1,3,3]
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
    
def remove_module_prefix(state_dict):
    """
    删除 'module.' 前缀，使 DataParallel 保存的模型能加载到非DataParallel模型上。
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_k] = v
    return new_state_dict

def rgb2gray(img):
    return 0.2989 * img[:, 0:1, :, :] + 0.5870 * img[:, 1:2, :, :] + 0.1140 * img[:, 2:3, :, :]

# === 掩码增强版 compute_dolp_aolp 函数 ===
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

class ProcessedS0Dataset(Dataset):
	def __init__(self, root_dir, mode='train'):
		self.root_dir = os.path.join(root_dir, 'processed_images', mode)
		self.files = os.listdir(self.root_dir)
		self.sample_keys = []
		self.mode = mode
		# f 文件名的字符串
		for f in self.files:
			if '_I0_' in f:
				if self.mode == 'train':
					parts = f.rsplit('_', 4)
					target = parts[0]
					turbidity = parts[2]
					rotate = parts[3]
					flip = parts[4].replace('.png', '')
					self.sample_keys.append((target, turbidity, rotate, flip))
				else:
					parts = f.rsplit('_', 2)
					target = parts[0]
					turbidity = parts[2].replace('.png', '')
					self.sample_keys.append((target, turbidity))
		# 图像转换并将值限制到[0,1]
		self.transform = torchvision.transforms.ToTensor()
	# 样本总数量
	def __len__(self):
		return len(self.sample_keys)
	# 根据索引获取样本
	def __getitem__(self, idx):
		if self.mode == 'train':
			target, turbidity, rotate, flip = self.sample_keys[idx]
		else:
			target, turbidity = self.sample_keys[idx]

		# 图象加载
		def load_img(direction):
			if self.mode == 'train':
				fname = f"{target}_{direction}_{turbidity}_{rotate}_{flip}.png"
			else:
				fname = f"{target}_{direction}_{turbidity}.png"
			fpath = os.path.join(self.root_dir, fname)
			return self.transform(Image.open(fpath).convert('RGB'))

		I0 = load_img('I0')
		I45 = load_img('I45')
		I90 = load_img('I90')
		GT0 = load_img('GT0')
		GT45 = load_img('GT45')
		GT90 = load_img('GT90')


		return {
			'I0': I0, 'I45': I45, 'I90': I90,
			'GT0': GT0, 'GT45': GT45, 'GT90': GT90,
			'target': target,
			'turbidity': turbidity,
			'rotate': rotate if self.mode == 'train' else -1,
			'flip': flip if self.mode == 'train' else -1
		}



if __name__ == '__main__':

	dicam = DICAM()

	dicam.to(device)

	mae_loss = nn.L1Loss()
	edge_loss_fn = EdgeLoss().to(device)

	vgg = Vgg19(requires_grad=False).to(device)
	train_set = ProcessedS0Dataset(root_dir=opt.root_dir, mode='train')

    
	# 优化器定义 其中w是正则化系数

	optim_g = optim.Adam(dicam.parameters(), lr=opt.learning_rate_g, betas=(opt.beta1, opt.beta2), weight_decay=opt.wd_g)
    

    
	initial_weights = [1.0 for _ in train_set.sample_keys]
	sampler = DynamicWeightedSampler(initial_weights, num_samples=len(initial_weights), replacement=True)
	train_loader = DataLoader(train_set, batch_size=opt.batch_size, sampler=sampler, num_workers=24, pin_memory=True)
    
	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optim_g,
		max_lr=opt.learning_rate_g,  # 一般为 1e-4
		steps_per_epoch=len(train_loader),
		epochs=100,  # 总轮数
		pct_start=0.3,  # 前 30% 的 epoch 用于升温，约 30 轮
		div_factor=1,  # 初始学习率 = max_lr / 1 = 1e-4，立即启用最大 lr
		final_div_factor=100,  # 最后学习率 = max_lr / 100 = 1e-6，平滑衰减
		anneal_strategy='cos'  # 使用余弦衰减策略
	)

	batches = len(train_loader)

	val_set = ProcessedS0Dataset(root_dir=opt.root_dir, mode='val')
	val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

	# 创建模型权重文件夹
	if not os.path.exists(opt.best_dir):
		os.makedirs(opt.best_dir)

	# 创建模型权重文件夹
	if not os.path.exists(opt.checkpoints_dir):
		os.makedirs(opt.checkpoints_dir)

	# 加载最新权重
	models_loaded = getLatestCheckpointName()
	latest_checkpoint = models_loaded

	ckpt_path = os.path.join(opt.checkpoints_dir, latest_checkpoint) if latest_checkpoint else None

	if latest_checkpoint is None or not os.path.exists(ckpt_path):
		print("🆕 No valid checkpoints found. Training from scratch.")
		start_epoch = 1
	else:
		print(f"📦 Loading model from checkpoint: {ckpt_path}")
		checkpoint_g = torch.load(ckpt_path)
		start_epoch = checkpoint_g['epoch'] + 1

		raw_state_dict = checkpoint_g['model_state_dict']


		# 工具函数：移除 DataParallel 的 'module.' 前缀
		def remove_module_prefix(state_dict):
			return {k.replace('module.', ''): v for k, v in state_dict.items()}


		# 判断是否含 module. 前缀
		has_module_prefix = any(k.startswith('module.') for k in raw_state_dict)

		model_to_load = dicam

		if isinstance(dicam, nn.DataParallel) and not has_module_prefix:
			print("⚠️ 当前模型使用 DataParallel，但权重不包含 module. 前缀，可能不匹配")
		if not isinstance(dicam, nn.DataParallel) and has_module_prefix:
			print("🧹 Removing 'module.' prefix from state_dict for single-GPU loading...")
			raw_state_dict = remove_module_prefix(raw_state_dict)

		load_result = model_to_load.load_state_dict(raw_state_dict, strict=True)
		print(
			f"✅ Model successfully restored (matched keys: {len(load_result.missing_keys)} missing, {len(load_result.unexpected_keys)} unexpected)")

		# 加载优化器状态（可选）
		if 'optimizer_state_dict' in checkpoint_g:
			optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])

		print('Restoring model from checkpoint ' + str(start_epoch))


	opt.best_val_loss = float('inf')
	for epoch in range(start_epoch, opt.end_epoch + 1):
		dicam.train()
		opt.total_mae_loss = 0.0
		opt.total_vgg_loss = 0.0
		opt.total_edge_loss = 0.0
		opt.total_loss = 0.0
		opt.total_ssim_loss = 0.0
		opt.total_color_loss = 0.0
		dynamic_weights = get_dynamic_turbidity_weights(epoch, max_epoch=opt.end_epoch)
		sample_weights = [dynamic_weights.get(str(t), 1.0) for (_, t, *_rest) in train_set.sample_keys]
		sampler.set_weights(sample_weights)

		for i_batch, sample_batched in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}', ncols=100)):
			I0 = sample_batched['I0'].to(device)
			I45 = sample_batched['I45'].to(device)
			I90 = sample_batched['I90'].to(device)
			S0 = torch.clamp((I0 + I90) / 2, 0.0, 1.0)
			GT0 = sample_batched['GT0'].to(device)
			GT45 = sample_batched['GT45'].to(device)
			GT90 = sample_batched['GT90'].to(device)
			GT_S0 = torch.clamp((GT0 + GT90) / 2, 0.0, 1.0)
            
            # 计算偏振特征 + 掩码
			DoLp, AoLP_1, AoLP_2, mask = compute_dolp_aolp(I0, I45, I90)
			S0 = ((I0 + I90) / 2).clamp(0.0, 1.0) * mask
			GT_S0 = ((GT0 + GT90) / 2).clamp(0.0, 1.0) * mask

			# 网络前向
			pred = dicam(DoLp, AoLP_1, AoLP_2, S0)
			pred = pred * mask

# 			DoLp, AoLP_1, AoLP_2 = compute_dolp_aolp(I0, I45, I90)

# 			pred = dicam(DoLp, AoLP_1, AoLP_2, S0)

			batch_mae_loss = torch.mul(opt.lambda_mae, mae_loss(pred, GT_S0))

			raw_ssim = ssim(pred, GT_S0, data_range=1, size_average=True)
			raw_ssim = torch.clamp(raw_ssim, 0.0, 1.0)
			batch_ssim_loss = 1 - raw_ssim

			batch_edge_loss = torch.mul(opt.lambda_edge, edge_loss_fn(pred, GT_S0))

			batch_color_loss = torch.mul(opt.lambda_color, color_stats_loss(pred, GT_S0))

			with torch.no_grad():
				vgg_clean = vgg(normalize_batch(GT_S0))
			vgg_pred = vgg(normalize_batch(pred))

			batch_vgg_loss = torch.mul(opt.lambda_vgg, mae_loss(vgg_pred.relu4_3, vgg_clean.relu4_3) )
			total_loss_tra = batch_mae_loss + batch_ssim_loss + batch_vgg_loss + batch_edge_loss + batch_color_loss
			total_loss_tra.backward()

			opt.batch_mae_loss = batch_mae_loss.detach().cpu().item()
			opt.total_mae_loss += opt.batch_mae_loss

			opt.batch_ssim_loss = batch_ssim_loss.detach().cpu().item()
			opt.total_ssim_loss += opt.batch_ssim_loss

			opt.batch_edge_loss = batch_edge_loss.detach().cpu().item()
			opt.total_edge_loss += opt.batch_edge_loss

			opt.batch_vgg_loss = batch_vgg_loss.detach().cpu().item()
			opt.total_vgg_loss += opt.batch_vgg_loss
			
			opt.batch_color_loss = batch_color_loss.detach().cpu().item()
			opt.total_color_loss += opt.batch_color_loss


			opt.batch_loss = opt.batch_mae_loss + opt.batch_ssim_loss + opt.batch_vgg_loss + opt.batch_edge_loss +  opt.batch_color_loss
			opt.total_loss += opt.batch_loss

			# 更新参数，清零防止累加
			optim_g.step()
			optim_g.zero_grad()

			del I0, I45, I90, GT0, GT45, GT90, S0,GT_S0
			del pred
			del vgg_pred
			del vgg_clean
			del raw_ssim

			print('\r Epoch : ' + str(epoch) + ' | (' + str(i_batch+1) + '/' + str(batches) + ') | l_mae: ' + str(opt.batch_mae_loss/2) + ' | l_ssim: ' + str(opt.batch_ssim_loss)+ ' | l_vgg: ' + str(opt.batch_vgg_loss)+ ' | l_edge: ' + str(opt.batch_edge_loss)+ ' | l_co: ' + str(opt.batch_color_loss), end='', flush=True)

		scheduler.step()
		print(f"📉 当前学习率: {get_lr(optim_g):.8f}")
		
		torch.cuda.empty_cache()

		print('\n\nFinished ep. %d, lr = %.6f, mean_mae = %.6f, mean_ssim = %.6f, mean_vgg = %.6f, mean_edge = %.6f, mean_color = %.6f' % (epoch, get_lr(optim_g), (opt.total_mae_loss / batches)/2,opt.total_ssim_loss / batches, opt.total_vgg_loss / batches ,opt.total_edge_loss / batches,opt.total_color_loss / batches))

		dicam.eval()
		os.makedirs(f'visual_results/epoch_{epoch}', exist_ok=True)
		with torch.no_grad():
			torch.cuda.empty_cache()
			opt.val_psnr_total = 0.0
			opt.val_ssim_total = 0.0
			opt.val_count = 0
			opt.val_total_mae_loss = 0.0
			opt.val_total_color_loss = 0.0
			opt.val_total_ssim_loss = 0.0
			opt.val_total_vgg_loss = 0.0
			opt.val_total_edge_loss = 0.0

			for i, val_batch in enumerate(tqdm(val_loader, desc=f'Validation Epoch {epoch}', ncols=100)):
				I0 = val_batch['I0'].to(device)
				I45 = val_batch['I45'].to(device)
				I90 = val_batch['I90'].to(device)
				S0 = torch.clamp((I0 + I90) / 2, 0.0, 1.0)
				GT0 = val_batch['GT0'].to(device)
				GT45 = val_batch['GT45'].to(device)
				GT90 = val_batch['GT90'].to(device)
				GT_S0 = torch.clamp((GT0 + GT90) / 2, 0.0, 1.0)

				# 3. 分别推理

# 				DoLp,AoLP_1, AoLP_2 = compute_dolp_aolp(I0, I45, I90)

# 				pred = dicam(DoLp, AoLP_1, AoLP_2, S0)
                
                 # 计算偏振特征 + 掩码
				DoLp, AoLP_1, AoLP_2, mask = compute_dolp_aolp(I0, I45, I90)
				S0 = ((I0 + I90) / 2).clamp(0.0, 1.0) * mask
				GT_S0 = ((GT0 + GT90) / 2).clamp(0.0, 1.0) * mask

				# 网络前向
				pred = dicam(DoLp, AoLP_1, AoLP_2, S0)
				pred = pred * mask

				val_mae = mae_loss(pred, GT_S0)
				opt.val_total_mae_loss += val_mae.item()

				val_color = color_stats_loss(pred, GT_S0)
				opt.val_total_color_loss += val_color.item()

				# SSIM Loss

				# 平均
				raw_ssim = ssim(pred, GT_S0, data_range=1, size_average=True)
				raw_ssim = torch.clamp(raw_ssim, 0.0, 1.0)

				ssim_loss_val = 1 - raw_ssim
				opt.val_total_ssim_loss += ssim_loss_val.item()


				# VGG Perceptual Loss
				# top
				vgg_gt = vgg(normalize_batch(GT_S0))
				vgg_pred = vgg(normalize_batch(pred))

				vgg_loss_val = mae_loss(vgg_pred.relu4_3, vgg_gt.relu4_3)


				opt.val_total_vgg_loss += vgg_loss_val.item()

				edge_loss_val = edge_loss_fn(pred, GT_S0)
				opt.val_total_edge_loss += edge_loss_val.item()


				# 4. 转 numpy
				def to_np(t): return np.clip(t.squeeze(0).cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)


				# top
				pred_np = to_np(pred)
				GT_np = to_np(GT_S0)



				# PSNR
				val_psnr = compare_psnr(GT_np, pred_np, data_range=1.0)
				# SSIM
				val_ssim = compare_ssim(GT_np, pred_np, channel_axis=2, data_range=1.0)


				# 汇总
				opt.val_psnr_total += val_psnr
				opt.val_ssim_total += val_ssim
				opt.val_count += 1

				target = val_batch['target'][0]
				turbidity = val_batch['turbidity'][0]

				pred_np = to_np(pred)
				gt_np = to_np(GT_S0)

				psnr_val = compare_psnr(gt_np, pred_np, data_range=1.0)
				ssim_val = compare_ssim(gt_np, pred_np, channel_axis=2, data_range=1.0)

				psnr_str = f"{psnr_val:.2f}"
				ssim_str = f"{ssim_val:.4f}"

				filename = f"{target}_{turbidity}_epoch{epoch}_psnr{psnr_str}_ssim{ssim_str}.png"
				path = os.path.join(f'visual_results/epoch_{epoch}', filename)
				os.makedirs(os.path.dirname(path), exist_ok=True)

				torchvision.utils.save_image(pred.squeeze(0).cpu(), path)

				# 清理显存
				del (
					I0, I45, I90, GT0, GT45, GT90,GT_np,S0,
					pred, pred_np,GT_S0,
					vgg_pred,path, filename,
					vgg_gt,psnr_val, ssim_val, val_psnr, val_ssim,
				) 
				torch.cuda.empty_cache()
			mean_mae = opt.val_total_mae_loss / opt.val_count
			mean_ssim = opt.val_total_ssim_loss / opt.val_count
			mean_vgg = opt.val_total_vgg_loss / opt.val_count
			mean_edge = opt.val_total_edge_loss / opt.val_count
			mean_color = opt.val_total_color_loss / opt.val_count
			mean_total_loss = (
					opt.lambda_mae * mean_mae +
					opt.lambda_vgg * mean_vgg +
					opt.lambda_edge * mean_edge +
					opt.lambda_color * mean_color +
					opt.lambda_ssim * mean_ssim  # 或 opt.lambda_ssim * mean_ssim
			)

			print(f"Validation Loss (MAE): {opt.lambda_mae * mean_mae:.6f} | SSIM: {opt.lambda_ssim * mean_ssim:.4f} | VGG: {opt.lambda_vgg * mean_vgg:.6f}| EDGE: {opt.lambda_edge * mean_edge:.6f} | COLOR: {opt.lambda_color * mean_color:.6f} | Total: {mean_total_loss:.6f}")
			print(f"Validation PSNR: {opt.val_psnr_total / opt.val_count: .2f} dB, SSIM: {opt.val_ssim_total / opt.val_count: .4f}")
			model_to_save = dicam
			torch.save({'epoch': epoch,
					'model_state_dict': model_to_save.state_dict(),
					'optimizer_state_dict': optim_g.state_dict(),
					'mae_loss': opt.total_mae_loss,
					'ssim_loss': opt.total_ssim_loss,
					'vgg_loss': opt.total_vgg_loss,
				    'edge_loss': opt.total_edge_loss,
					'color_loss': opt.total_color_loss,
					'opt': opt,
					'total_loss': opt.total_loss,
					# ⬇️ 保存验证损失（分项 + 总）
					'val_mae_loss': mean_mae,
					'val_ssim_loss': mean_ssim,
					'val_vgg_loss': mean_vgg,
					'val_edge_loss': mean_edge,
					'val_color_loss': mean_color,
					'val_total_loss': mean_total_loss,
					'val_psnr': opt.val_psnr_total / opt.val_count,
					'val_ssim': opt.val_ssim_total / opt.val_count},
				   os.path.join(opt.checkpoints_dir, f'DICAM_{epoch}.pt'))
			print(f"✅ 模型保存成功: ")

			# === 保存最佳模型 ===
			if epoch == start_epoch or mean_total_loss < opt.best_val_loss:
				opt.best_val_loss = mean_total_loss
				best_model_path = os.path.join(opt.best_dir,  f'DICAM_best_epoch{epoch}_loss{mean_total_loss:.6f}.pth')
				torch.save({
					'epoch': epoch,
					'model_state_dict': model_to_save.state_dict(),
					'optimizer_state_dict': optim_g.state_dict(),
					'mae_loss': opt.total_mae_loss,
					'ssim_loss': opt.total_ssim_loss,
					'vgg_loss': opt.total_vgg_loss,
					'edge_loss': opt.total_edge_loss,
					'opt': opt,
					'total_loss': opt.total_loss,
					'val_mae_loss': mean_mae,
					'val_color_loss': mean_color,
					'val_ssim_loss': mean_ssim,
					'val_vgg_loss': mean_vgg,
					'val_edge_loss': mean_edge,
					'val_total_loss': mean_total_loss,
					'val_psnr': opt.val_psnr_total / opt.val_count,
					'val_ssim': opt.val_ssim_total / opt.val_count,
				}, best_model_path)
				print("💾 最佳模型已更新并保存：DICAM_best.pth")
                




