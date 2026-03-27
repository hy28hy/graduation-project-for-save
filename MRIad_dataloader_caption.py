import json
import cv2
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from data.nsa import patch_ex

# 医学影像的标准化参数（如果不需要 ImageNet 色彩空间，可以在 main 函数里调整）
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class BraTSDataset_caption(Dataset):
    def __init__(self, type, root, caption_path="./data/brats_captions_Qwen3-VL.json", template_prob=0.2):
        self.type = type
        self.root = root
        self.template_prob = template_prob
        self.split = 'train' if type == 'train' else 'valid'
        self.image_size = (256, 256)

        # 1. 加载 Caption JSON
        self.captions = {}
        if self.type == 'train':
            if os.path.exists(caption_path):
                print(f"Loading captions from {caption_path}...")
                with open(caption_path, 'r') as f:
                    self.captions = json.load(f)
            else:
                print(f"Warning: Caption file not found at {caption_path}")

        # 2. 构建数据列表（区分正常与异常，不区分物品类别）
        self.data = []
        self.good_indices = [] 
        
        for condition in ['good', 'Ungood']:
            img_dir = os.path.join(self.root, self.split, condition, 'img')
            if not os.path.exists(img_dir):
                continue
                
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith(('.png', '.jpg')):
                    rel_path = f"{self.split}/{condition}/img/{fname}"
                    is_anomaly = 1 if condition == 'Ungood' else 0
                    
                    item = {
                        'filename': rel_path,
                        'abs_path': os.path.join(img_dir, fname),
                        'is_anomaly': is_anomaly 
                    }
                    
                    if is_anomaly == 1:
                        item['maskname'] = os.path.join(self.root, self.split, condition, 'label', fname)
                    
                    idx = len(self.data)
                    self.data.append(item)
                    
                    if is_anomaly == 0:
                        self.good_indices.append(idx)
                        
        print(f"Loaded {len(self.data)} images for {self.type} set.")

    def __len__(self):
        return len(self.data)

    def find_idx(self, idx):
        possible_indices = [i for i in self.good_indices if i != idx]
        if not possible_indices:
            raise ValueError("No possible normal image found for NSA.")
        return random.choice(possible_indices)

    def get_nsa_args(self):
        return {'width_bounds_pct': ((0.03, 0.4), (0.03, 0.4)),
                'num_patches': 4}

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- NSA 数据增强逻辑 (仅对正常样本) ---
        do_nsa = False
        if self.type == 'train' and item['is_anomaly'] == 0 and idx % 2 == 0:
            nsa_idx = self.find_idx(idx)
            nsa_item = self.data[nsa_idx]
            do_nsa = True
        else:
            nsa_item = item

        # --- 图像读取 ---
        transform_fn = transforms.Resize(self.image_size)

        target = cv2.imread(nsa_item['abs_path'])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = Image.fromarray(target, "RGB")
        target = transform_fn(target)

        source = cv2.imread(item['abs_path'])
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = Image.fromarray(source, "RGB")
        source = transform_fn(source)

        # --- Mask 生成 ---
        if do_nsa:
            source_np, mask_np = patch_ex(np.asarray(target), np.asarray(source), **self.get_nsa_args())
            mask = (mask_np[:, :, 0] * 255.0).astype(np.uint8)
            source = Image.fromarray(source_np, "RGB")
        else:
            if item.get('maskname') and os.path.exists(item['maskname']):
                mask = cv2.imread(item['maskname'], cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.uint8) * 255
            else:
                mask = np.zeros(self.image_size, dtype=np.uint8)

        target = transforms.ToTensor()(target)
        source = transforms.ToTensor()(source)

        # --- 直接生成 MRI 专属 Prompt ---
        prompt = ""
        if self.type == 'train':
            if random.random() < self.template_prob:
                prompt = "A detailed magnetic resonance image of a human brain."
            elif item['filename'] in self.captions:
                prompt = self.captions[item['filename']]
            else:
                prompt = "A detailed magnetic resonance image of a human brain."
        else:
            prompt = "A detailed magnetic resonance image of a human brain."

        mask = Image.fromarray(mask, "L")
        mask = transforms.ToTensor()(mask)

        normalize_fn = transforms.Normalize(mean=mean_train, std=std_train)
        source = normalize_fn(source)
        target = normalize_fn(target)

        # 【重点修改】保留 clsname 和 label 的键名以防主脚本报错，但赋予统一的常量值
        return dict(
            jpg=target, 
            txt=prompt, 
            hint=source, 
            mask=mask, 
            filename=item['filename'], 
            clsname='brain_mri',  # 占位符，保证字典结构不破裂
            label=0               # 强制归一为 0 类，代替原来的 image_idx
        )