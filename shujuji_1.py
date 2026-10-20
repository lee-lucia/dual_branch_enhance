import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class PolarizationDataset(Dataset):
    def __init__(self, root_dir, mode='train', split_ratio=(0.7, 0.2, 0.1), seed=42, augment=True, val_targets=None, test_targets=None):
        self.root_dir = root_dir
        self.mode = mode  # 'train' / 'val' / 'test'
        self.turbidity_levels = ['1.00', '1.76', '2.52', '3.28', '4.04']
        self.turbidity_values = [float(t) for t in self.turbidity_levels]
        self.directions = ['0', '45', '90']
        self.augment = augment
        self.val_targets = val_targets
        self.test_targets = test_targets


        VALID_SUFFIXES = ['.png']
        VALID_DIRECTIONS = self.directions

        self.all_data = []
        self.missing_count = 0

        gt_dir = os.path.join(root_dir, 'gt')
        all_files = os.listdir(gt_dir)

        filtered_files = []
        for f in all_files:
            if not any(f.endswith(suffix) for suffix in VALID_SUFFIXES):
                continue
            if '_' not in f:
                continue
            parts = f.rsplit('_', 1)
            if len(parts) != 2:
                continue
            direction_part = parts[1].split('.')[0]
            if direction_part not in VALID_DIRECTIONS:
                continue
            filtered_files.append(f)

        all_targets = sorted(list(set(f.rsplit('_', 1)[0] for f in filtered_files)))
        print(f"Found {len(filtered_files)} valid .png files for directions {VALID_DIRECTIONS}")
        print(f"Extracted {len(all_targets)} unique targets")

        if self.test_targets is not None:
            test = self.test_targets
            remaining = [t for t in all_targets if t not in test]
        else:
            remaining, test = train_test_split(all_targets, test_size=split_ratio[2], random_state=seed)

        if self.val_targets is not None:
            val = self.val_targets
            train = [t for t in remaining if t not in val]
        else:
            train, val = train_test_split(remaining, test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]),
                                          random_state=seed)

        if mode == 'train':
            self.targets = train
        elif mode == 'val':
            self.targets = val
        elif mode == 'test':
            self.targets = test
        else:
            raise ValueError("mode must be one of ['train', 'val', 'test']")

        print(f"[{self.mode}] Loaded {len(self.targets)} targets. Example: {self.targets[:5]}")

        # 保存划分结果
        save_split_dir = os.path.join(self.root_dir, 'splits')
        os.makedirs(save_split_dir, exist_ok=True)

        save_path = os.path.join(save_split_dir, f'{self.mode}_targets.txt')
        with open(save_path, 'w') as f:
            for target in self.targets:
                f.write(target + '\n')

        print(f"[{self.mode}] Target list saved to {save_path}")

        self.pil_to_tensor = transforms.ToTensor()

        self._prepare_data()

    def center_crop_1024x1024(self, img_tensor):
        C, H, W = img_tensor.shape
        top = (H - 1024) // 2
        left = (W - 1024) // 2
        return img_tensor[:, top:top + 1024, left:left + 1024]

    def resize_256(self, img_tensor):
        pil_img = transforms.ToPILImage()(img_tensor)
        resized_img = pil_img.resize((256, 256), Image.BICUBIC)
        return transforms.ToTensor()(resized_img)

    def _prepare_data(self):
        patch_count = 0
        total_steps = len(self.turbidity_levels) * len(self.targets)
        progress_bar = tqdm(total=total_steps, desc=f"🔄 Preparing [{self.mode}] data")

        # 创建保存处理后图像的文件夹（根据mode决定文件夹）
        save_dir = os.path.join(self.root_dir, 'processed_images', self.mode)  # 根据mode创建不同文件�??
        os.makedirs(save_dir, exist_ok=True)  # 创建目标文件夹（train、val、test�?

        for turbidity_str, turbidity_float in zip(self.turbidity_levels, self.turbidity_values):
            for target in self.targets:
                input_imgs, gt_imgs = [], []

                for d in self.directions:
                    in_path = os.path.join(self.root_dir, turbidity_str, f"{target}_{d}.png")
                    gt_path = os.path.join(self.root_dir, 'gt', f"{target}_{d}.png")
                    try:
                        input_img = self.pil_to_tensor(Image.open(in_path).convert('RGB'))
                        gt_img = self.pil_to_tensor(Image.open(gt_path).convert('RGB'))

                        # 所有数据集都裁剪为中心1024x1024并resize到256x256
                        input_img = self.center_crop_1024x1024(input_img)
                        gt_img = self.center_crop_1024x1024(gt_img)
                        input_img = self.resize_256(input_img)
                        gt_img = self.resize_256(gt_img)

                        input_imgs.append(input_img)
                        gt_imgs.append(gt_img)
                    except FileNotFoundError as e:
                        print(f"Missing image for {target} at turbidity {turbidity_str}, direction {d}: {e}")

                if len(input_imgs) != 3 or len(gt_imgs) != 3:
                    print(f"Skipping incomplete group: {target} - {turbidity_str}")
                    self.missing_count += 1
                    progress_bar.update(1)
                    continue

                input_tensor = torch.stack(input_imgs, dim=0)  # [3, 3, H, W]
                gt_tensor = torch.stack(gt_imgs, dim=0)  # [3, 3, H, W]

                if self.mode == 'train':
                    patch_group = {
                        'I0': input_tensor[0],
                        'I45': input_tensor[1],
                        'I90': input_tensor[2],
                        'GT0': gt_tensor[0],
                        'GT45': gt_tensor[1],
                        'GT90': gt_tensor[2],
                        'target': target,
                        'turbidity': turbidity_float,
                    }

                    if self.augment:
                        for rotate in [0, 1, 2, 3]:
                            for flip in [None, 'h', 'v']:
                                augmented = {}
                                for k in ['I0', 'I45', 'I90', 'GT0', 'GT45', 'GT90']:
                                    img = patch_group[k]
                                    patch = torch.rot90(img, rotate, [1, 2])
                                    if flip == 'h':
                                        patch = torch.flip(patch, [2])
                                    elif flip == 'v':
                                        patch = torch.flip(patch, [1])
                                    augmented[k] = patch

                                augmented['target'] = patch_group['target']
                                augmented['turbidity'] = patch_group['turbidity']

                                for direction, patch in augmented.items():
                                    if direction.startswith('I') or direction.startswith('GT'):
                                        patch_image = transforms.ToPILImage()(patch)
                                        file_name = f"{patch_group['target']}_{direction}_{patch_group['turbidity']}_r{rotate}_f{flip if flip else 'n'}.png"
                                        patch_image.save(os.path.join(save_dir, file_name))

                                self.all_data.append(augmented)
                                patch_count += 1
                    else:
                        for direction in ['I0', 'I45', 'I90', 'GT0', 'GT45', 'GT90']:
                            patch_image = transforms.ToPILImage()(patch_group[direction])
                            file_name = f"{patch_group['target']}_{direction}_{patch_group['turbidity']}_orig.png"
                            patch_image.save(os.path.join(save_dir, file_name))
                else:  # For test and validation, don't crop or patch, just save the images
                    # 提取每个方向的图像（input_tensor形状为[3, C, H, W]
                    input_patches = [input_tensor[i] for i in range(3)]  # [I0, I45, I90]
                    gt_patches = [gt_tensor[i] for i in range(3)]  # [GT0, GT45, GT90]
                    all_patches = input_patches + gt_patches  # 合并输入和GT

                    for direction, patch in zip(['I0', 'I45', 'I90', 'GT0', 'GT45', 'GT90'], all_patches):
                        patch_image = transforms.ToPILImage()(patch)
                        file_name = f"{target}_{direction}_{turbidity_float}.png"
                        patch_image.save(os.path.join(save_dir, file_name))

                progress_bar.update(1)

        progress_bar.close()
        print(f"Finished preparing data. Skipped {self.missing_count} incomplete image groups.")
        print(f"Total patches generated: {patch_count}")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        sample = self.all_data[idx]
        return {
            'I0': sample['I0'],
            'I45': sample['I45'],
            'I90': sample['I90'],
            'GT0': sample['GT0'],
            'GT45': sample['GT45'],
            'GT90': sample['GT90'],
            'target': sample['target'], 
            'turbidity': sample['turbidity'],
        }



root_dir = "/root/autodl-fs/60_final"

val_targets = ['coin', 'coral', 'cube','live', 'motor', 'neu','pineapple','ruler','wheel','tea']
test_targets = ['color_card', 'coral','milk', 'shell1']

set1 = PolarizationDataset(root_dir=root_dir, mode='train', val_targets=val_targets, test_targets=test_targets)
set2 = PolarizationDataset(root_dir=root_dir, mode='val', val_targets=val_targets, test_targets=test_targets)
set3 = PolarizationDataset(root_dir=root_dir, mode='test', val_targets=val_targets, test_targets=test_targets)




