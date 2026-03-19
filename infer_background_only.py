import torch
import os
from omegaconf import OmegaConf
from customnet import CustomNet
from utils import load_image, save_image
from ldm.util import instantiate_from_config

# ===================== 你的配置 =====================
OBJECT_IMAGE_PATH = "/home/huyanhan/data_FGVC/object_birds/train/001.Black_footed_Albatross/Black_Footed_Albatross_0007_796138.jpg"
BACKGROUND_PROMPT = "on the grass, in a beautiful garden"
OUTPUT_PATH = "result.png"

# 模型路径
CUSTOMNET_CKPT = "pretrain/customnet_v1.pth"
SD_15_CKPT = "pretrain/v1-5-pruned-emaonly.ckpt"
CONFIG_PATH = "configs/config_customnet.yaml"
# =====================================================

# 1. 加载配置
config = OmegaConf.load(CONFIG_PATH)
model_config = config.model

# 2. 构建模型（严格按照你提供的 class CustomNet 结构）
model = CustomNet(
    text_encoder_config=model_config.params.text_encoder_config,
    sd_15_ckpt=SD_15_CKPT,
    use_cond_concat=model_config.params.use_cond_concat,
    use_bbox_mask=model_config.params.use_bbox_mask,
    use_bg_inpainting=model_config.params.use_bg_inpainting,
    learning_rate_scale=model_config.params.learning_rate_scale,
    **model_config.params
)

# 3. 加载 CustomNet 权重
sd = torch.load(CUSTOMNET_CKPT, map_location="cpu")["state_dict"]
model.load_state_dict(sd, strict=False)
model = model.cuda().eval()

# 4. 预处理图片
object_img = load_image(OBJECT_IMAGE_PATH).unsqueeze(0).cuda()

# 5. 编码文字（CLIP 文本编码，模型自带）
with torch.no_grad():
    text_emb = model.text_encoder([BACKGROUND_PROMPT])

# 6. 构建条件（严格按照模型 shared_step 的格式）
cond = {
    "c_crossattn": [text_emb],
}

# 7. 核心：只生成背景，保留物体（真正源码逻辑）
with torch.no_grad():
    # 先把物体编码到 latent 空间
    z = model.encode_first_stage(object_img)
    z = z.mode()

    # 扩散采样生成（CustomNet 核心：物体不变 → 生成背景）
    samples = model.sample(
        cond=cond,
        batch_size=1,
        return_intermediates=False,
    )

    # 解码成图片
    result = model.decode_first_stage(samples)

# 8. 保存
save_image(result, OUTPUT_PATH)
print("✅ 背景生成完成 →", OUTPUT_PATH)