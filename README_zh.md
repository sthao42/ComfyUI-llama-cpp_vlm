# ComfyUI-llama-cpp
在 ComfyUI 中基于 llama.cpp 框架原生运行 LLM & VLM 模型。  
**[[📃English](./README.md)]**   

## 预览
![](./img/preview.jpg) 

## 安装步骤

### 📋 环境要求
> [!IMPORTANT]
> 本节点套件需要 **`llama-cpp-python` v0.3.48 或更高版本**。预编译的 CUDA 13.0、CUDA 12.8 及 Metal 加速 Wheel 文件可以在 [`JamePeng/llama-cpp-python`](https://github.com/JamePeng/llama-cpp-python) 仓库中获取。

#### 安装节点:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sthao42/ComfyUI-llama-cpp_vlm.git
python -m pip install -r ComfyUI-llama-cpp_vlm/requirements.txt
```

### 模型路径:
- 请将下载的 `.gguf` 模型放置在 `ComfyUI/models/LLM` 目录中.  

	> 在使用VLM模型进行图像推理之前, 请确保已经下载并选择了主模型对应的`mmproj`权重文件.

## 致谢
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
