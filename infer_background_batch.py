import os
import torch
from customnet import CustomNet
from utils import load_image, save_image

# 路径
OBJECT_DIR = "your_multi_view_objects/"  # 你的多视角无背景图文件夹
BACKGROUND_PROMPT = "on the table, photorealistic, 8k"
OUTPUT_DIR = "fused_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
model = CustomNet().cuda().eval()
model.load_pretrain("pretrain/customnet_v1.pth")

# 批量处理
for fname in os.listdir(OBJECT_DIR):
    path = os.path.join(OBJECT_DIR, fname)
    
    img = load_image(path).cuda()
    
    with torch.no_grad():
        out = model.generate_with_background(
            image=img,
            background_prompt=BACKGROUND_PROMPT
        )
    
    save_image(out, os.path.join(OUTPUT_DIR, fname))
    print(f"✅ 处理完成：{fname}")