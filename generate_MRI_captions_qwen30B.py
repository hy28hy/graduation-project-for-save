import os
import sys
import json
import argparse
import glob
from tqdm import tqdm
from vllm import LLM, SamplingParams


# -----------------------------------------------------------------------------
# 1. Environment & Path Setup
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# -----------------------------------------------------------------------------
# 2. Prompt Strategy
# -----------------------------------------------------------------------------
# 这里保留了你之前用的 pd 模式的 prompt
DEFAULT_PROMPT = (
    "Describe the visual features of image in detail with a focus on the main object, with around 60 words and maybe 4 or 5 sentences."
)

# -----------------------------------------------------------------------------
# 3. Core Functions
# -----------------------------------------------------------------------------

def load_model(model_path, tp_size=4, use_fp8=True):
    print(f"\n🚀 Preparing to load Qwen3-VL from: {model_path}")
    print(f"⚡ Strategy: Tensor Parallelism = {tp_size}")
    
    quant_config = "fp8" if use_fp8 else None
    print(f"📉 Quantization: {quant_config if quant_config else 'None (BF16/Original)'}")

    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            quantization=quant_config, 
            gpu_memory_utilization=0.95, 
            max_model_len=4096, 
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
            allowed_local_media_path="/"
        )
        print("✅ vLLM Engine loaded successfully.")
        
        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.8,
            max_tokens=128,
            stop_token_ids=[151643, 151645] 
        )
        return llm, sampling_params
    except Exception as e:
        print(f"\n❌ Model load failed: {e}")
        sys.exit(1)

def get_image_files(root_dir):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_list = []
    
    print(f"📂 Scanning sets at: {root_dir}")
    if not os.path.isdir(root_dir):
        print(f"❌ Error: Data path is not a directory: {root_dir}")
        return []

    # 使用递归查找所有文件
    search_path = os.path.join(root_dir, '**', '*.*')
    all_files = glob.glob(search_path, recursive=True)
    
    for abs_path in all_files:
        if os.path.splitext(abs_path)[1].lower() in image_extensions:
            # 过滤掉路径中包含 'label' 的文件夹（通常是掩码 mask）
            if 'label' in abs_path.split(os.sep):
                continue
            
            # 生成相对路径作为 JSON 的 key
            rel_path = os.path.relpath(abs_path, root_dir)
            image_list.append((abs_path, rel_path))
                        
    print(f"✅ Found {len(image_list)} valid brain images (excluding labels).")
    return image_list

def generate_caption_vllm(llm, sampling_params, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"file://{image_path}"
                    }
                }, 
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]

    outputs = llm.chat(messages, sampling_params, use_tqdm=False)
    generated_text = outputs[0].outputs[0].text
    return generated_text

# -----------------------------------------------------------------------------
# 4. Main Loop
# -----------------------------------------------------------------------------

def main():
    # 更新为你实际的服务器路径
    default_model_path = "/data2/chenxuwu/zihaowan_workplace/qwen/Qwen3-VL-235B-Instruct" 
    default_output_path = "/data2/chenxuwu/zihaowan_workplace/dataset/brats_captions_Qwen3-VL.json"
    default_data_path = "/data2/chenxuwu/zihaowan_workplace/dataset/BraTS2021_slice"
    
    parser = argparse.ArgumentParser(description="Generate captions using Qwen with vLLM")
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Dataset root path")
    parser.add_argument("--output", type=str, default=default_output_path, help="Output JSON path")
    parser.add_argument("--model_path", type=str, default=default_model_path, help="Local model path")
    parser.add_argument("--tp", type=int, default=4, help="Tensor Parallelism size (GPU count)")
    parser.add_argument("--no_fp8", action="store_true", help="Disable FP8 quantization")
    
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data path not found: {args.data_path}")
        sys.exit(1)
    
    output_dir = os.path.dirname(args.output)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    # --- Resume Logic ---
    captions_dict = {}
    if os.path.exists(args.output):
        print(f"🔄 Found existing output file: {args.output}")
        try:
            with open(args.output, 'r') as f:
                captions_dict = json.load(f)
            print(f"   Resuming... Loaded {len(captions_dict)} existing captions.")
        except json.JSONDecodeError:
            print("⚠️  JSON corrupted. Starting over.")
            captions_dict = {}

    all_images = get_image_files(args.data_path)
    # 过滤已处理的图片
    todo_images = [item for item in all_images if item[1] not in captions_dict]
    
    print(f"📝 To generate: {len(todo_images)}")
    if len(todo_images) == 0:
        print("🎉 All done!")
        return

    # 加载模型 (vLLM)
    use_fp8 = not args.no_fp8
    llm, sampling_params = load_model(args.model_path, tp_size=args.tp, use_fp8=use_fp8)

    print("\n🚀 Starting generation...")
    save_interval = 20 
    
    try:
        for i, (abs_path, rel_path) in enumerate(tqdm(todo_images, desc="Captioning")):
            try:
                caption = generate_caption_vllm(llm, sampling_params, abs_path)
                captions_dict[rel_path] = caption
            except Exception as e:
                print(f"\n❌ Error on {rel_path}: {e}")
                captions_dict[rel_path] = "<ERROR_GENERATING>"
                continue
            
            if (i + 1) % save_interval == 0:
                with open(args.output, 'w') as f:
                    json.dump(captions_dict, f, indent=4)
                    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted! Saving currently processed data...")
    
    with open(args.output, 'w') as f:
        json.dump(captions_dict, f, indent=4)
    print(f"\n✅ Done! Saved to {args.output}")

if __name__ == "__main__":
    main()