import sys
import os
sys.path.append(os.getcwd())
from share import *
from utils.util import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# 👇 1. 替换为我们刚才写的 BraTS 数据加载器
from data.MRIad_dataloader_caption import BraTSDataset_caption 

from cdm.model import create_model
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import argparse

# --- 核心限制 ---
cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    setup_seed(args.seed)

    # 2. 更新日志命名逻辑 (移除了 setting 参数)
    log_name = f'brats_template_prob{args.template_prob}_{args.vlm_model}'
    save_dir = f'./incre_val/{log_name}/'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🚀 Foundation Model Training (BraTS MRI)")
    print(f"   Save Dir: {save_dir}")
    print(f"   Resume From: {args.resume_path}")
    print(f"   VLM Model: {args.vlm_model}")
    print(f"   Template Prob: {args.template_prob}")
    print(f"{'='*80}\n")

    print(f">>> 正在加载 BraTS 数据集...")

    # 👇 3. 实例化 BraTS 数据集 (不再返回元组，直接返回 Dataset 对象)
    train_dataset = BraTSDataset_caption(
        type='train', 
        root=args.data_path, 
        caption_path=args.caption_path,
        template_prob=args.template_prob
    )
    test_dataset = BraTSDataset_caption(
        type='test', 
        root=args.data_path, 
        caption_path=args.caption_path,
        template_prob=args.template_prob
    )

    # 4. 初始化模型 (如果 MRI 有专属的 yaml 配置文件，请在这里更改)
    model = create_model('models/cdad_mvtec.yaml').cpu()
    model.learning_rate = args.learning_rate

    # 5. 加载初始权重
    if not os.path.exists(args.resume_path):
        raise FileNotFoundError(f"未找到初始权重: {args.resume_path}，请先运行 build_base_model.py")
        
    print(f">>> 加载初始权重: {args.resume_path}")
    weights = torch.load(args.resume_path)
    select_weights = {key: weights[key] for key in weights if not 'control_model' in key} 
    model.load_state_dict(select_weights, strict=False)
    
    print(">>> [Mode] Full Fine-tuning (无 LoRA，全量更新)")

    # 6. 配置 Logger 和 Checkpoint
    model.set_log_name(f'{log_name}/foundation')
    
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=log_name, version="foundation")
    csv_logger = CSVLogger(save_dir="lightning_logs", name=log_name, version="foundation")

    ckpt_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=save_dir,
        filename='foundation_model', 
        mode='max',
        save_top_k=1
    )

    strategy = None
    if args.gpus and len(args.gpus) > 1:
        strategy = 'ddp'

    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        gpus=args.gpus, 
        precision=32, 
        strategy=strategy,
        callbacks=[ckpt_callback],
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
        max_epochs=args.max_epoch,
        check_val_every_n_epoch=args.check_v,
        enable_progress_bar=False 
    )

    # 👇 7. 直接传入 Dataset 对象，不需要索引 [task_id]
    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)

    # 8. 开始训练
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    
    print(f"\n✅ Foundation Model 训练完成！")
    print(f"   保存位置: {os.path.join(save_dir, 'foundation_model.ckpt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Foundation Model for BraTS MRI")

    parser.add_argument("--resume_path", default='./models/base.ckpt', help="由 build_base_model.py 生成的初始权重")
    
    # 👇 更新默认路径为 BraTS 实际路径
    parser.add_argument("--data_path", default="/data2/chenxuwu/zihaowan_workplace/dataset/BraTS2021_slice", type=str)
    parser.add_argument("--caption_path", default="/data2/chenxuwu/zihaowan_workplace/dataset/brats_captions_Qwen3-VL.json", type=str)
    
    parser.add_argument("--vlm_model", default="Qwen3-VL-30B", type=str, help="用于标记实验日志名称")
    parser.add_argument("--template_prob", default=0.2, type=float, help="Probability of using simple template instead of detailed caption")

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float) 
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--check_v", default=25, type=int)
    parser.add_argument("--gpus", nargs='+', type=int, default=[4,5,6,7], help="GPU IDs")

    args = parser.parse_args()

    main(args)