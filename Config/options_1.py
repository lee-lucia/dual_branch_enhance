import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoints_dir', default='/root/autodl-fs/7.4/polarcanshu')
parser.add_argument('--best_dir', default='/root/autodl-fs/7.4/best')

parser.add_argument('--root_dir', default='/root/autodl-fs/60_final')


parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--learning_rate_g', type=float, default=1e-04)
parser.add_argument('--best_val_loss', type=float, default=float('inf'))

parser.add_argument('--end_epoch', type=int, default=60)
parser.add_argument('--img_extension', default='.png')
parser.add_argument('--image_size', type=int ,default=256)

parser.add_argument('--beta1', type=float ,default=0.9)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.00005)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_mae_loss', type=float, default=0.0)
parser.add_argument('--total_mae_loss', type=float, default=0.0)

parser.add_argument('--batch_ssim_loss', type=float, default=0.0)
parser.add_argument('--total_ssim_loss', type=float, default=0.0)

parser.add_argument('--batch_edge_loss', type=float, default=0.0)
parser.add_argument('--total_edge_loss', type=float, default=0.0)

parser.add_argument('--batch_vgg_loss', type=float, default=0.0)
parser.add_argument('--total_vgg_loss', type=float, default=0.0)

parser.add_argument('--batch_color_loss', type=float, default=0.0)
parser.add_argument('--total_color_loss', type=float, default=0.0)

parser.add_argument('--batch_loss', type=float, default=0.0)
parser.add_argument('--total_loss', type=float, default=0.0)

parser.add_argument('--val_psnr_total', type=float, default=0.0)
parser.add_argument('--val_ssim_total', type=float, default=0.0)
parser.add_argument('--val_count', type=float, default=0.0)


parser.add_argument('--val_total_mae_loss', type=float, default=0.0)
parser.add_argument('--val_total_ssim_loss', type=float, default=0.0)
parser.add_argument('--val_total_vgg_loss', type=float, default=0.0)
parser.add_argument('--val_total_edge_loss', type=float, default=0.0)
parser.add_argument('--val_total_color_loss', type=float, default=0.0)


parser.add_argument('--lambda_mae', type=float, default=1.0)
parser.add_argument('--lambda_vgg', type=float, default=0.5)
parser.add_argument('--lambda_ssim', type=float, default=1.0)
parser.add_argument('--lambda_edge', type=float, default=0.5)
parser.add_argument('--lambda_color', type=float, default=1.0)

parser.add_argument('--testing_epoch', type=int, default=1)
parser.add_argument('--testing_mode', default="Nat")

opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)
