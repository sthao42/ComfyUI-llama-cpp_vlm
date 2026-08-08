# ComfyUI-llama-cpp_vlm

An advanced, high-performance ComfyUI custom node suite for running Large Language Models (LLMs) and Vision-Language Models (VLMs) natively based on [`llama.cpp`](https://github.com/ggerganov/llama.cpp) and [`llama-cpp-python`](https://github.com/JamePeng/llama-cpp-python).

**[[📃 中文版](./README_zh.md)]**

---

## 🌟 Key Features in This Fork

### 🖼️ Dynamic Multi-Image & Tagging Workflows
* **Auto-Expanding Input Sockets (`image_0`, `image_1`, ...):** Starts clean with a single `image_0` socket. As you connect `Load Image` nodes, sockets dynamically expand up to `image_8`, automatically pruning unlinked trailing ports when disconnected.
* **Inline Tagging & Placeholder Interleaving:** Reference specific images directly in your custom or system prompts using tags like `<Picture 0>`, `<Picture 1>`, `<image_0>`, `<image_1>`, etc. The node automatically places base64 visual payload tokens at the exact placeholder positions.
* **Multi-Image Preset Prompts:** Includes built-in prompts for multi-image visual comparison, difference analysis, and text-to-video / image reference generation.

### 🤖 Broad VLM & LLM Model Support
* **Google Gemma 4 (12B & 31B):** Native support for `Gemma4` and `Gemma4-Thinking` with automatic non-causal attention batching (`n_ubatch >= 2048`) to prevent `llama.cpp` assertion crashes.
* **Qwen 3.6 & Qwen 3.5 Series:** Full support for `Qwen3.6`, `Qwen3.6-Thinking`, `Qwen3.5`, `Qwen3.5-Thinking`, `Qwen3-VL`, `Qwen3-VL-Thinking`, and `Qwen2.5-VL` with hybrid context cache clearing.
* **OCR & Specialized Handlers:** Support for `DeepSeek-OCR`, `MinerU2.5-Pro`, `MiniCPM-v4.5 / v4.6`, `PaddleOCR-VL-1.5`, `Qwen3-ASR`, `Step3-VL`, `GLM-4.6V`, `LFM2.5-VL`, `Granite-Docling`, and more.

### ⚡ Performance & Long-Context Optimizations
* **Flash Attention (`flash_attn`):** Reduces KV cache VRAM consumption by 50–70% on long context windows (8k–128k) and accelerates multi-image token processing.
* **Multi-Token Prediction (MTP / Speculative Decoding):** Toggleable `enable_mtp` using `LlamaNGramMapDecoding` for accelerated, human-readable text generation without raw token ID artifacts.
* **Uncapped Output Tokens (`max_tokens = -1`):** Generate long detailed captions, code, or prompts up to `4096` tokens or set `max_tokens = -1` for uncapped generation up to the remaining context window limit (`n_ctx`).
* **Batch Tuning (`n_batch`, `n_ubatch`):** Fine-grained control over prompt evaluation batch size and physical micro-batch sizes.
* **Smart Context Protection:** Automatically calculates estimated token usage (`system_prompt` + `user_prompt` + `N_images * 1024`) against `n_ctx` and logs console warnings if context length needs scaling.
* **Backward & Forward `mmproj` Compatibility:** Dynamically inspects `llama-cpp-python` handler parameters for both `mmproj_path` and `clip_model_path`.

---

## 📸 Preview

![](./img/preview.jpg)

---

## 📦 Installation

### 1. Clone the repository into ComfyUI custom nodes:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sthao42/ComfyUI-llama-cpp_vlm.git
python -m pip install -r ComfyUI-llama-cpp_vlm/requirements.txt
```

### 2. Download Models:
* Place GGUF model files in `ComfyUI/models/LLM`.
* Place matching vision projector (`mmproj`) files in `ComfyUI/models/LLM`.

---

## 🛠️ Usage & Parameter Reference

### `llama_cpp_model_loader`
* **`n_ctx`**: Total context length limit (e.g. `8192`, `16384`, `32768`).
* **`vram_limit`**: VRAM limit in GB (`-1` = no limit).
* **`n_batch` / `n_ubatch`**: Logical prompt batch and physical micro-batch size.
* **`flash_attn`**: Toggle Flash Attention for low VRAM long-context processing.
* **`offload_kqv`**: Offload Key/Query/Value KV cache tensors directly to GPU VRAM for maximum speed.
* **`kv_cache_type`**: Choose `"f16"`, `"q8_0"`, or `"q4_0"`. Quantizing KV cache saves up to 75% VRAM on 16k–128k context windows.
* **`n_threads`**: CPU thread tuning for generation (`0` = auto-detect).
* **`enable_mtp`**: Enable Multi-Token Prediction (speculative decoding).

### `llama_cpp_instruct_adv`
* **`image_0`**: Initial image socket. Connecting an image automatically reveals `image_1`, `image_2`, up to `image_7`.
* **`custom_prompt`**: Input user prompt. Use `<Picture 0>`, `<Picture 1>` placeholders to position images inline.
* **`preset_prompt`**: Select pre-configured prompts for captioning, tagging, Midjourney/Flux prompt styling, or multi-image comparison.

### `llama_cpp_parameters`
* **`max_tokens`**: Maximum generated tokens (default: `4096`, set `-1` for uncapped generation).
* **`stop`**: Comma-separated list of phrases to halt generation instantly (e.g. `"###, \n\n"`).
* **`dry_multiplier` / `dry_base` / `dry_allowed_length`**: DRY (Don't Repeat Yourself) sampler to eliminate repetitive phrasing.
* **`dynatemp_range`**: Dynamic temperature sampling range (`0.0` = disabled).
* **`xtc_threshold` / `xtc_probability`**: Exclude Top Choices (XTC) sampler.
* **`reasoning_budget`**: Max token budget for thinking models (`Gemma4-Thinking`, `Qwen3.6-Thinking`).
* **`temperature`**, **`top_p`**, **`min_p`**, **`repeat_penalty`**, **`present_penalty`**.

---

## 📜 Credits & Acknowledgments
* [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) by @JamePeng
* [ComfyUI-llama-cpp](https://github.com/lihaoyun6/ComfyUI-llama-cpp) by @lihaoyun6
* [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) by @kijai
* [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by @comfyanonymous
