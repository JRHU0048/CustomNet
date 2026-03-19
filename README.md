# 🎯 目标
**指定的物体图片数据集**，想要**批量处理**，让每张图都自动生成：
1. **抠出物体**（去掉背景）
2. **生成物体多角度**（3D 视角）
3. **生成不同背景**（草地、房间、桌面…）
4. **物体自然融入背景**（最终合成图）


# 🚀 第一步：环境安装
```bash
conda create -n customnet python=3.10 -y
conda activate customnet

pip install setuptools==59.8.0
pip install albumentations
pip install -r requirements.txt
```


# 🚀 第二步：安装依赖工具
```bash
cd ~
git clone https://gitclone.com/github.com/CompVis/taming-transformers.git
cd taming-transformers
pip install -e .
cd ~/CustomNet


cd ~
git clone git+https://github.com/openai/CLIP.git@main#egg=clip
cd CLIP
pip install -e .
cd ~/CustomNet
```


# 🚀 第二步：下载依赖工具
进入 `extralibs` 克隆 3 个工具模型：
```bash
cd extralibs
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
git clone https://github.com/salesforce/LAVIS.git
git clone https://github.com/cvlab-columbia/zero123.git
cd ..
```

# 三个模型的作用：
- **FastSAM**：抠物体
- **LAVIS (BLIP2)**：给图片生成文字描述
- **Zero123**：生成物体多角度（3D）

# 🚀 第三步：放入你自己的数据集
**创建数据集文件夹**
```
examples/data/
    ├── 001.jpg
    ├── 002.jpg
    ├── 003.jpg
    └── ...
```

CustomNet 会**自动批量处理这个文件夹里的所有图片**。

# 🚀 第四步：一键批处理 → 自动完成 4 步
官方提供了**一键全自动脚本**：
**抠图 → 生成caption → 生成多角度 → 生成背景融合图**

直接运行：
```bash
sh scripts/process_data.sh
```

## 这个脚本会自动做 4 件事
### 1. 物体抠图（FastSAM）
输入：你的原图
输出：**只保留物体，去掉背景**

### 2. 生成物体文字描述（LAVIS / BLIP2）
输入：你的物体图
输出：`a golden cat` 这类 caption

### 3. 生成物体多角度（Zero123）
输入：物体图
输出：**前视、侧视、俯视** 等多个视角

### 4. 生成背景 + 物体融合（Stable Diffusion + CustomNet）
输入：物体多角度 + 文字背景描述
输出：**物体完美融入新背景**


# 🚀 第五步：运行推理 → 批量生成最终图
下载官方权重：
```bash
pretrain/customnet_v1.pth
```

运行 gradio 批量生成：
```bash
sh scripts/run_app.sh
```

**支持批量处理文件夹！**

# 🎯 最终你会得到什么？
对**你数据集中的每一张物体图**，都会输出：

1. 抠好的物体图（无背景）
2. 物体多角度图（3D）
3. 物体 + 任意背景融合图
4. 位置、光照、比例全部自然

**完全满足你需求：多视角 + 多背景 + 批量处理**

# 📌 最简总结
```bash
conda activate customnet
cd extralibs && git clone ...  # 安装3个工具
把你的图片放进 examples/data/
sh scripts/process_data.sh     # 一键批处理
sh scripts/run_app.sh         # 批量生成最终图
```
