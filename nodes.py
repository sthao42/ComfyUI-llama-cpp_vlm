import os
import io
import gc
import re
import json
import base64
import random
import inspect
import torch

import numpy as np
from PIL import Image, ImageDraw
try:
    from .support.cqdm import cqdm
    from .support.gguf_layers import get_layer_count
    from .support import prompt_enhancer_preset as preset_mod
except ImportError:
    from support.cqdm import cqdm
    from support.gguf_layers import get_layer_count
    from support import prompt_enhancer_preset as preset_mod

import folder_paths
import comfy.model_management as mm
import comfy.utils

from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler
)

try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    _MTMD = True
except Exception:
    _MTMD = False

chat_handlers = ["None", "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha", "MiniCPM-v2.6"]

# Pre-compiled module-level Regular Expressions for performance optimization
PLACEHOLDER_PATTERN = re.compile(r'(?:<|\[)(?:Picture|image|img)\s*([0-9]\d*)(?:>|\])', re.IGNORECASE)
THINK_BLOCK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
THINK_BLOCK_UNCLOSED_PATTERN = re.compile(r'<think>.*$', re.DOTALL)
JSON_CODEBLOCK_PATTERN = re.compile(r'^```(?:json)?\s*|\s*```$', re.IGNORECASE | re.MULTILINE)

def gaussian_filter_2d(image_array: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 2D Gaussian blur filter on a NumPy float32 array without scipy dependency."""
    if sigma <= 0:
        return image_array
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image_array, sigma=sigma)
    except ImportError:
        radius = int(round(3.0 * sigma))
        if radius < 1:
            return image_array
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()

        res = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=image_array)
        res = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=res)
        return res.astype(np.float32)

def get_safe_model_path(base_dir: str, filename: str) -> str:
    """Validate and sanitize model filenames against directory traversal attacks."""
    if not filename or filename == "None":
        return ""
    full_path = os.path.abspath(os.path.join(base_dir, filename))
    base_dir_abs = os.path.abspath(base_dir)
    if not full_path.startswith(base_dir_abs):
        raise ValueError(f"Security Alert: Directory traversal detected in model path: '{filename}'")
    return full_path

if _MTMD:
    chat_handlers.append("DeepSeek-OCR")

try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
    chat_handlers += ["Gemma3"]
except Exception:
    Gemma3ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
    chat_handlers += ["Gemma4", "Gemma4-Thinking"]
except Exception:
    Gemma4ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    chat_handlers += ["Qwen2.5-VL", "MinerU2.5-Pro"]
except Exception:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    chat_handlers += ["Qwen3-VL", "Qwen3-VL-Thinking"]
except Exception:
    Qwen3VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen38ChatHandler
except Exception:
    Qwen38ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
except Exception:
    Qwen35ChatHandler = None

if Qwen38ChatHandler is not None or Qwen35ChatHandler is not None:
    chat_handlers += [
        "Qwen3.5", "Qwen3.5-Thinking",
        "Qwen3.6", "Qwen3.6-Thinking",
        "Qwen3.8", "Qwen3.8-Thinking"
    ]

try:
    from llama_cpp.llama_chat_format import (GLM46VChatHandler, LFM2VLChatHandler, GLM41VChatHandler)
    chat_handlers += ["GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "LFM2-VL"]
except Exception:
    GLM46VChatHandler = None
    LFM2VLChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM25VLChatHandler
    chat_handlers += ["LFM2.5-VL"]
except Exception:
    LFM25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
    chat_handlers += ["Granite-Docling"]
except Exception:
    GraniteDoclingChatHandler = None

try:
    from llama_cpp.llama_chat_format import MiniCPMv45ChatHandler
    chat_handlers += ["MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"]
except Exception:
    MiniCPMv45ChatHandler = None

try:
    from llama_cpp.llama_chat_format import MiniCPMv46ChatHandler
    chat_handlers += ["MiniCPM-v4.6", "MiniCPM-v4.6-Thinking"]
except Exception:
    MiniCPMv46ChatHandler = None

try:
    from llama_cpp.llama_chat_format import PaddleOCRChatHandler
    chat_handlers += ["PaddleOCR-VL-1.5"]
except Exception:
    PaddleOCRChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3ASRChatHandler
    chat_handlers += ["Qwen3-ASR"]
except Exception:
    Qwen3ASRChatHandler = None

try:
    from llama_cpp.llama_chat_format import Step3VLChatHandler
    chat_handlers += ["Step3-VL"]
except Exception:
    Step3VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import GenericMTMDChatHandler
    chat_handlers += ["Generic-MTMD"]
except Exception:
    GenericMTMDChatHandler = None

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    messages = {}
    sys_prompts = {}

    @classmethod
    def clean_state(cls, state_id: int = -1, **kwargs):
        target_id = kwargs.get("id", state_id)
        if target_id == -1:
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            cls.messages.pop(f"{target_id}", None)
            cls.sys_prompts.pop(f"{target_id}", None)

    @classmethod
    def clean(cls, clear_all: bool = False):
        if cls.llm is not None:
            try:
                cls.llm.close()
            except Exception:
                pass

        if cls.chat_handler is not None and hasattr(cls.chat_handler, "_exit_stack"):
            try:
                cls.chat_handler._exit_stack.close()
            except Exception:
                pass

        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        if clear_all:
            cls.clean_state()

        gc.collect()
        mm.soft_empty_cache()
    
    @classmethod
    def load_model(cls, config):
        def get_chat_handler(chat_handler):
            match chat_handler:
                case "Qwen3.5"|"Qwen3.5-Thinking"|"Qwen3.6"|"Qwen3.6-Thinking"|"Qwen3.8"|"Qwen3.8-Thinking":
                    return Qwen38ChatHandler if Qwen38ChatHandler is not None else Qwen35ChatHandler
                case "Qwen3-VL"|"Qwen3-VL-Thinking":
                    return Qwen3VLChatHandler
                case "Qwen3-ASR":
                    return Qwen3ASRChatHandler
                case "Qwen2.5-VL"|"MinerU2.5-Pro":
                    return Qwen25VLChatHandler
                case "LLaVA-1.5":
                    return Llava15ChatHandler
                case "LLaVA-1.6":
                    return Llava16ChatHandler
                case "Moondream2":
                    return MoondreamChatHandler
                case "nanoLLaVA":
                    return NanoLlavaChatHandler
                case "llama3-Vision-Alpha":
                    return Llama3VisionAlphaChatHandler
                case "MiniCPM-v2.6":
                    return MiniCPMv26ChatHandler
                case "MiniCPM-v4.5"|"MiniCPM-v4.5-Thinking":
                    return MiniCPMv45ChatHandler
                case "MiniCPM-v4.6"|"MiniCPM-v4.6-Thinking":
                    return MiniCPMv46ChatHandler
                case "Gemma3":
                    return Gemma3ChatHandler
                case "Gemma4"|"Gemma4-Thinking":
                    return Gemma4ChatHandler
                case "GLM-4.6V"|"GLM-4.6V-Thinking":
                    return GLM46VChatHandler
                case "GLM-4.1V-Thinking":
                    return GLM41VChatHandler
                case "LFM2-VL":
                    return LFM2VLChatHandler
                case "LFM2.5-VL":
                    return LFM25VLChatHandler
                case "Granite-Docling":
                    return GraniteDoclingChatHandler
                case "DeepSeek-OCR":
                    return MTMDChatHandler
                case "PaddleOCR-VL-1.5":
                    return PaddleOCRChatHandler
                case "Step3-VL":
                    return Step3VLChatHandler
                case "Generic-MTMD":
                    return GenericMTMDChatHandler
                case "None":
                    return None
                case _:
                    raise ValueError(f'Unknown model type: "{chat_handler}"')
        
        cls.clean(clear_all=True)
        cls.current_config = config.copy()
        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler = config["chat_handler"]
        n_ctx = config["n_ctx"]
        vram_limit = config["vram_limit"]
        image_max_tokens = config["image_max_tokens"]
        image_min_tokens = config["image_min_tokens"]
        n_batch = config.get("n_batch", 2048)
        n_ubatch = config.get("n_ubatch", 512)
        enable_mtp = config.get("enable_mtp", False)
        n_gpu_layers = -1

        # Auto-adjust n_ubatch and n_batch for multimodal models (Qwen, Gemma, GLM, MiniCPM, LFM) to ensure image token chunks (up to 2048 tokens) fit in physical micro-batches.
        if (mmproj and mmproj != "None") or chat_handler != "None" or "gemma" in model.lower() or "qwen" in model.lower():
            min_ubatch = min(2048, n_ctx)
            if n_ubatch < min_ubatch:
                print(f"[llama-cpp_vlm] Multimodal handler ({chat_handler}) active. Auto-adjusting n_ubatch from {n_ubatch} to {min_ubatch} for image token evaluation.")
                n_ubatch = min_ubatch
            if n_batch < n_ubatch:
                n_batch = n_ubatch
        
        llm_dir = os.path.join(folder_paths.models_dir, 'LLM')
        model_path = get_safe_model_path(llm_dir, model)
        handler = get_chat_handler(chat_handler)
        
        if vram_limit != -1:
            gguf_layers = get_layer_count(model_path) or 32
            gguf_size = os.path.getsize(model_path) * 1.55 / (1024 ** 3)
            gguf_layer_size = gguf_size / gguf_layers
        
        if mmproj and mmproj != "None":
            mmproj_path = get_safe_model_path(llm_dir, mmproj)
            if chat_handler == "None":
                raise ValueError('"chat_handler" cannot be None!')
            
            if vram_limit != -1:
                mmproj_size = os.path.getsize(mmproj_path)  * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit - mmproj_size) / gguf_layer_size))
            
            print(f"[llama-cpp_vlm] Loading clip:  {mmproj}")
            
            handler_params = inspect.signature(handler.__init__).parameters
            think_mode = "Thinking" in chat_handler
            kwargs = {"verbose": False}
            if "mmproj_path" in handler_params:
                kwargs["mmproj_path"] = mmproj_path
            else:
                kwargs["clip_model_path"] = mmproj_path
            if "chat_format" in handler_params:
                kwargs["chat_format"] = None
            if chat_handler in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                kwargs["force_reasoning"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
            elif any(name in chat_handler for name in ["MiniCPM-v4.5", "MiniCPM-v4.6", "GLM-4.6V", "GLM-4.1V", "Qwen3.5", "Qwen3.6", "Qwen3.8", "Gemma4"]):
                kwargs["enable_thinking"] = think_mode

            if _MTMD:
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens

            try:
                cls.chat_handler = handler(**kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\nPlease update llama-cpp-python from 'https://github.com/JamePeng/llama-cpp-python/releases'")

        else:
            if vram_limit != -1:
                n_gpu_layers = max(1, int(vram_limit / gguf_layer_size))
            if handler is not None:
                cls.chat_handler = handler(verbose=False)
            else:
                cls.chat_handler = None
        
        print(f"[llama-cpp_vlm] Loading model: {model}")
        print(f"[llama-cpp_vlm] n_gpu_layers = {n_gpu_layers}")
        llama_init_params = inspect.signature(Llama.__init__).parameters
        
        llama_kwargs = {
            "model_path": model_path,
            "chat_handler": cls.chat_handler,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "verbose": False
        }
        
        if config.get("flash_attn", True):
            if "flash_attn" in llama_init_params:
                llama_kwargs["flash_attn"] = True
            elif "flash_attn_type" in llama_init_params:
                llama_kwargs["flash_attn_type"] = 1

        if config.get("offload_kqv", True) and "offload_kqv" in llama_init_params:
            llama_kwargs["offload_kqv"] = True

        kv_cache_type = config.get("kv_cache_type", "f16")
        if kv_cache_type != "f16":
            type_map = {"f16": 1, "q4_0": 2, "q8_0": 8}
            kv_val = type_map.get(kv_cache_type, 1)
            if "type_k" in llama_init_params:
                llama_kwargs["type_k"] = kv_val
            if "type_v" in llama_init_params:
                llama_kwargs["type_v"] = kv_val

        n_threads = config.get("n_threads", 0)
        if n_threads > 0 and "n_threads" in llama_init_params:
            llama_kwargs["n_threads"] = n_threads
            
        enable_mtp = config.get("enable_mtp", False)
        speculative_mode = config.get("speculative_mode", "auto")
        draft_model = config.get("draft_model", "None")
        draft_model_path = ""
        if draft_model and draft_model != "None":
            draft_model_path = get_safe_model_path(llm_dir, draft_model)

        has_spec = enable_mtp or (speculative_mode != "auto") or bool(draft_model_path)
        if has_spec:
            try:
                from llama_cpp.llama_speculative import SpecConfig, SpeculativeType
                resolved_type = None

                if speculative_mode == "DFlash":
                    resolved_type = getattr(SpeculativeType, "DRAFT_DFLASH", None)
                elif speculative_mode == "DSpark":
                    resolved_type = getattr(SpeculativeType, "DRAFT_DSPARK", None)
                elif speculative_mode == "MTP":
                    resolved_type = getattr(SpeculativeType, "DRAFT_MTP", None)
                elif speculative_mode == "NGRAM":
                    resolved_type = getattr(SpeculativeType, "NGRAM_MAP_K", None)
                elif speculative_mode == "auto":
                    if draft_model_path:
                        dm_lower = draft_model.lower()
                        if "dspark" in dm_lower:
                            resolved_type = getattr(SpeculativeType, "DRAFT_DSPARK", SpeculativeType.DRAFT_DFLASH)
                        elif "mtp" in dm_lower:
                            resolved_type = getattr(SpeculativeType, "DRAFT_MTP", SpeculativeType.DRAFT_DFLASH)
                        else:
                            resolved_type = getattr(SpeculativeType, "DRAFT_DFLASH", getattr(SpeculativeType, "DRAFT_MTP", SpeculativeType.NGRAM_MAP_K))
                    elif enable_mtp:
                        resolved_type = SpeculativeType.NGRAM_MAP_K

                if resolved_type is not None and resolved_type != SpeculativeType.NONE:
                    spec_kwargs = {"spec_type": resolved_type}
                    if draft_model_path and resolved_type in {
                        getattr(SpeculativeType, "DRAFT_DFLASH", None),
                        getattr(SpeculativeType, "DRAFT_DSPARK", None),
                        getattr(SpeculativeType, "DRAFT_MTP", None),
                        getattr(SpeculativeType, "DRAFT_SIMPLE", None),
                    }:
                        spec_kwargs["draft_model_path"] = draft_model_path
                        spec_kwargs["draft_n_max"] = 8

                    if "speculative" in llama_init_params:
                        llama_kwargs["speculative"] = SpecConfig(**spec_kwargs)
                        print(f"[llama-cpp_vlm] Speculative Decoding enabled using SpecConfig ({resolved_type.name})"
                              f"{f' with draft model: {draft_model}' if draft_model_path else ''}.")
                        if (mmproj and mmproj != "None") and draft_model_path:
                            print(f"[llama-cpp_vlm] Note: Draft model '{draft_model}' is active alongside multimodal clip '{mmproj}'. "
                                  "llama.cpp draft sidecars accelerate text generation and are automatically bypassed when processing image inputs.")
                    else:
                        from llama_cpp.llama_speculative import LlamaNGramMapDecoding
                        llama_kwargs["draft_model"] = LlamaNGramMapDecoding(spec_type=resolved_type)
                        print(f"[llama-cpp_vlm] Speculative Decoding enabled using LlamaNGramMapDecoding ({resolved_type.name}).")
            except Exception as e:
                # Fallback for older llama-cpp-python versions
                try:
                    from llama_cpp.llama_speculative import LlamaNGramMapDecoding
                    llama_kwargs["draft_model"] = LlamaNGramMapDecoding()
                    print(f"[llama-cpp_vlm] Speculative Decoding fallback to legacy LlamaNGramMapDecoding: {e}")
                except Exception as e2:
                    print(f"[llama-cpp_vlm] Warning: Speculative decoding failed to initialize: {e} / {e2}")

        cls.llm = Llama(**llama_kwargs)

any_type = AnyType("*")

if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean(clear_all=True)
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    print("[llama-cpp_vlm] Model cleanup hook applied!")

llm_extensions = ['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.gguf']
folder_paths.folder_names_and_paths["LLM"] = ([os.path.join(folder_paths.models_dir, "LLM")], llm_extensions)
preset_prompts = {
    "Empty - Nothing": "",
    "Normal - Describe": "Describe this @.",
    "Prompt Style - Tags": "Your task is to generate a clean list of comma-separated tags for a text-to-@ AI, based *only* on the visual information in the @. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
    "Prompt Style - Simple": "Analyze the @ and generate a simple, single-sentence text-to-@ prompt. Describe the main subject and the setting concisely.",
    "Prompt Style - Detailed": "Generate a detailed, artistic text-to-@ prompt based on the @. Combine the subject, their actions, the environment, lighting, and overall mood into a single, cohesive paragraph of about 2-3 sentences. Focus on key visual details.",
    "Prompt Style - Extreme Detailed": "Generate an extremely detailed and descriptive text-to-@ prompt from the @. Create a rich paragraph that elaborates on the subject's appearance, textures of clothing, specific background elements, the quality and color of light, shadows, and the overall atmosphere. Aim for a highly descriptive and immersive prompt.",
    "Prompt Style - Cinematic": "Act as a master prompt engineer. Create a highly detailed and evocative prompt for an @ generation AI. Describe the subject, their pose, the environment, the lighting, the mood, and the artistic style (e.g., photorealistic, cinematic, painterly). Weave all elements into a single, natural language paragraph, focusing on visual impact.",
    "Creative - Detailed Analysis": "Describe this @ in detail, breaking down the subject, attire, accessories, background, and composition into separate sections.",
    "Creative - Summarize Video": "Summarize the key events and narrative points in this video.",
    "Creative - Short Story": "Write a short, imaginative story inspired by this @ or video.",
    "Creative - Refine & Expand Prompt": "Refine and enhance the following user prompt for creative text-to-@ generation. Keep the meaning and keywords, make it more expressive and visually rich. Output **only the improved prompt text itself**, without any reasoning steps, thinking process, or additional commentary.",
    "Multi-Image - Compare & Describe": "Compare the provided images (<Picture 1>, <Picture 2>, etc.). Describe the visual differences, subjects, styles, and composition across each picture.",
    "Multi-Image - Reference Video Prompt": "Analyze the reference images (<Picture 1>, <Picture 2>, etc.) and generate a cohesive text-to-video / text-to-image prompt detailing the progression, subject consistency, and scene dynamics across the frames.",
    "Vision - *Bounding Box": 'Locate every instance that belongs to the following categories: "#". Report bbox coordinates in {"bbox_2d": [x1, y1, x2, y2], "label": "string"} JSON format as a List.'
}
preset_tags = list(preset_prompts.keys())

def image2base64(image):
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

def parse_json(json_str):
    if isinstance(json_str, (dict, list)):
        return json_str
    if not json_str:
        raise ValueError("JSON string is empty.")
    json_output = JSON_CODEBLOCK_PATTERN.sub("", str(json_str).strip())
    try:
        parsed = json.loads(json_output)
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed

def scale_image(image: torch.Tensor, max_size: int = 128):
    if hasattr(image, "ndim") and image.ndim == 4:
        image = image.squeeze(0) if image.shape[0] == 1 else image[0]
    img_np = np.clip(255.0 * (image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)), 0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    img_pil = Image.fromarray(img_np)
    
    w, h = img_pil.size
    scale = min(max_size / max(w, h), 1.0) if max(w, h) > 0 else 1.0
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return np.array(img_resized)

def qwen3bbox(image, json):
    if hasattr(image, "ndim") and image.ndim == 4:
        image = image.squeeze(0) if image.shape[0] == 1 else image[0]
    img_np = np.clip(255.0 * (image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)), 0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    img = Image.fromarray(img_np)
    bboxes = []
    for item in json:
        if not isinstance(item, dict) or "bbox_2d" not in item:
            continue
        x0, y0, x1, y1 = item["bbox_2d"]
        size = 1000
        x0 = x0 / size * img.width
        y0 = y0 / size * img.height
        x1 = x1 / size * img.width
        y1 = y1 / size * img.height
        bboxes.append((x0, y0, x1, y1))
    return bboxes

def draw_bbox(image, json, mode):
    label_colors = {}
    if hasattr(image, "ndim") and image.ndim == 4:
        image = image.squeeze(0) if image.shape[0] == 1 else image[0]
    img_np = np.clip(255.0 * (image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)), 0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    
    for item in json:
        if not isinstance(item, dict):
            continue
        try:
            label = item["label"]
        except Exception:
            try:
                label = item["text_content"]
            except Exception:
                label = "bbox"
        if "bbox_2d" not in item:
            continue
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
            size = 1000
            x0 = x0 / size * img.width
            y0 = y0 / size * img.height
            x1 = x1 / size * img.width
            y1 = y1 / size * img.height
        bbox = (x0, y0, x1, y1)
        
        if label not in label_colors:
            label_colors[label] = tuple(random.randint(80, 180) for _ in range(3))
        color = label_colors[label]
        draw.rectangle(bbox, outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), str(label))
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), str(label), fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

def strip_think_block(text: str) -> str:
    """Sanitize output text by stripping reasoning <think>...</think> blocks."""
    if not text:
        return ""
    cleaned = THINK_BLOCK_PATTERN.sub('', text)
    cleaned = THINK_BLOCK_UNCLOSED_PATTERN.sub('', cleaned)
    return cleaned.strip()

def _flatten_image_tensors(val) -> list:
    results = []
    if val is None:
        return results
    if isinstance(val, list):
        for item in val:
            results.extend(_flatten_image_tensors(item))
    elif hasattr(val, "ndim") and val.ndim == 4:
        for i in range(val.shape[0]):
            results.append(val[i])
    else:
        results.append(val)
    return results

def collect_image_inputs(kwargs: dict) -> list:
    """Collect image and video frame inputs dynamically from kwargs (image_0..image_8 and video_0..video_8)."""
    all_images = []
    keys_to_check = (
        ["images"]
        + [f"image_{i}" for i in range(9)]
        + ["video", "videos"]
        + [f"video_{i}" for i in range(9)]
    )
    for key in keys_to_check:
        img_val = kwargs.get(key, None)
        if img_val is not None:
            all_images.extend(_flatten_image_tensors(img_val))
                
    return all_images

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = folder_paths.get_filename_list("LLM")
        model_list = [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"]+[f for f in all_llms if "mmproj" in f.lower()]
        draft_model_list = ["None"] + [f for f in all_llms if "mmproj" not in f.lower()]
            
        return {"required": {
            "model": (model_list,),
            "mmproj": (mmproj_list, {"default": "None"}),
            "chat_handler": (chat_handlers, {"default": "None"}),
            "n_ctx": ("INT", {
                "default": 16384,
                "min": 1024, "max": 327680, "step": 128,
                "tooltip": "Context length limit."
            }),
            "vram_limit": ("INT", {
                "default": -1,
                "min": -1, "max": 1024, "step": 1,
                "tooltip": "VRAM usage limit in GB (-1 = no limit)\nReference range; actual usage may slightly exceed."
            }),
            "image_min_tokens": ("INT", {
                "default": 1024,
                "min": 0, "max": 16384, "step": 32,
                "tooltip": "Minimum image tokens allocated per image patch."
            }),
            "image_max_tokens": ("INT", {
                "default": 4096,
                "min": 0, "max": 32768, "step": 32,
                "tooltip": "Maximum image tokens allocated per image patch."
            }),
            "n_batch": ("INT", {
                "default": 2048,
                "min": 128, "max": 327680, "step": 64,
                "tooltip": "Logical batch size for processing prompt tokens."
            }),
            "n_ubatch": ("INT", {
                "default": 512,
                "min": 128, "max": 327680, "step": 64,
                "tooltip": "Physical micro-batch size. Must be >= prompt/image tokens for non-causal attention models like Gemma."
            }),
            "enable_mtp": ("BOOLEAN", {
                "default": False,
                "tooltip": "Enable Multi-Token Prediction (MTP / Speculative Decoding) using SpecConfig (NGRAM or DFlash/DSpark draft sidecars) to accelerate token generation."
            }),
            "flash_attn": ("BOOLEAN", {
                "default": True,
                "tooltip": "Enable Flash Attention for memory reduction (KV cache) and faster long-context multi-image processing."
            }),
            "offload_kqv": ("BOOLEAN", {
                "default": True,
                "tooltip": "Offload Key/Query/Value KV cache tensors directly to GPU VRAM for maximum inference speed."
            }),
            "kv_cache_type": (["f16", "q8_0", "q4_0"], {
                "default": "f16",
                "tooltip": "KV cache quantization type. Quantizing KV cache to q8_0 or q4_0 reduces VRAM usage for long context windows by up to 75%."
            }),
            "n_threads": ("INT", {
                "default": 0,
                "min": 0, "max": 128, "step": 1,
                "tooltip": "CPU threads for token generation (0 = auto-detect)."
            }),
            },
            "optional": {
                "speculative_mode": (["auto", "NGRAM", "DFlash", "DSpark", "MTP"], {
                    "default": "auto",
                    "tooltip": "Speculative decoding mode. 'auto' selects DFlash/MTP if draft_model is provided, or NGRAM if enable_mtp is True."
                }),
                "draft_model": (draft_model_list, {
                    "default": "None",
                    "tooltip": "Optional draft / sidecar model for DFlash / DSpark / MTP speculative decoding (llama.cpp 0.3.49+).\nNote: Draft sidecars accelerate text generation and are automatically bypassed when processing image inputs."
                }),
            }
        }

    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"
    
    def loadmodel(self, model, mmproj="None", chat_handler="None", n_ctx=16384, vram_limit=-1, image_min_tokens=1024, image_max_tokens=4096, n_batch=2048, n_ubatch=512, enable_mtp=False, flash_attn=True, offload_kqv=True, kv_cache_type="f16", n_threads=0, speculative_mode="auto", draft_model="None"):
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler": chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "enable_mtp": enable_mtp,
            "flash_attn": flash_attn,
            "offload_kqv": offload_kqv,
            "kv_cache_type": kv_cache_type,
            "n_threads": n_threads,
            "speculative_mode": speculative_mode,
            "draft_model": draft_model
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            print("[llama-cpp_vlm] Loading model...")
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (custom_config,)

class llama_cpp_instruct_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "preset_prompt": (preset_tags, {"default": preset_tags[1]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": 'user_prompt\n\nFor preset hints marked with an "*", this will be used to fill the placeholder (e.g., Object names in BBox detection)\nOtherwise, this will override the preset prompts.'}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "inference_mode": (["one by one", "images", "video"], {
                    "default": "one by one",
                    "tooltip": "one by one: Read one image at a time\nimages:  \tRead all images at once\nvideo:  \tTreat the input images as video"
                }),
                "max_frames": ("INT", {
                    "default": 24,
                    "min": 2,
                    "max": 1024,
                    "step": 1,
                    "tooltip": 'Number of frames to sample evenly from input video.\n(for "video" mode only)'
                }),
                "max_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 16384,
                    "step": 64,
                    "tooltip": 'Max size of input images in "images" and "video" modes.'
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "control_after_generate": True}),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the model after inference."
                }),
                "save_states": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve the context of this conversation in RAM."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "parameters": ("LLAMACPPARAMS",),
                "image_0": ("IMAGE",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "video_0": ("IMAGE",),
                "queue_handler": (any_type, {"tooltip": "Used to control the execution order of instruct nodes."}),
            },
            
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "output_list", "state_uid")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    @classmethod
    def IS_CHANGED(cls, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, parameters=None, queue_handler=None, **kwargs):
        if seed is None or seed == -1:
            return float("nan")
        return f"{seed}_{save_states}"

    def sanitize_seed(self, seed, offset=0):
        if seed is None or seed == -1:
            val = random.randint(0, 0x7fffffff)
        else:
            val = (int(seed) + offset) & 0xFFFFFFFF
        if val == 0xFFFFFFFF:
            val = 0xFFFFFFFF - 1
        return val

    def sanitize_messages(self, messages):
        clean_messages = []
        for msg in messages:
            msg_copy = {"role": msg.get("role", "user")}
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"}
                        })
                    elif isinstance(item, dict):
                        new_content.append(item.copy())
                    else:
                        new_content.append(item)
                msg_copy["content"] = new_content
            else:
                msg_copy["content"] = content
            clean_messages.append(msg_copy)
        return clean_messages
    
    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, parameters=None, queue_handler=None, **kwargs):
        base_seed = seed
        active_seed = self.sanitize_seed(base_seed)

        if not LLAMA_CPP_STORAGE.llm:
            LLAMA_CPP_STORAGE.load_model(llama_model)
        
        if parameters is None:
            parameters = {}
            
        _uid = parameters.get("state_uid", None)
        _parameters = parameters.copy()
        _parameters.pop("state_uid", None)
        if _parameters.get("max_tokens") in (-1, 0):
            _parameters.pop("max_tokens", None)
            
        uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
        
        last_sys_prompt = LLAMA_CPP_STORAGE.sys_prompts.get(f"{uid}", None)
        video_input = inference_mode == "video"
        system_prompts = "请将输入的图片序列当做视频而不是静态帧序列, " + system_prompt if video_input else system_prompt
        if last_sys_prompt != system_prompts:
            messages = []
            LLAMA_CPP_STORAGE.clean_state(state_id=uid)
            LLAMA_CPP_STORAGE.sys_prompts[f"{uid}"] = system_prompts
            if system_prompts.strip():
                messages.append({"role": "system", "content": system_prompts})
        else:
            if save_states:
                try:
                    print(f"[llama-cpp_vlm] Loading state and history id={uid}...")
                    messages = LLAMA_CPP_STORAGE.messages.get(f"{uid}", [])
                except Exception as e:
                    messages = []
            else:
                messages = []
        out1 = ""
        out2 = []
        user_content = []
        prompt_text = ""
        if custom_prompt.strip() and "*" not in preset_prompt:
            prompt_text = custom_prompt.strip()
        else:
            prompt_text = preset_prompts[preset_prompt].replace("#", custom_prompt.strip()).replace("@", "video" if video_input else "image")
            
        # Collect all image inputs from image_0..image_7 (and images if provided)
        all_images = collect_image_inputs(kwargs)

        placeholders = list(PLACEHOLDER_PATTERN.finditer(prompt_text))
        has_zero = any(int(m.group(1)) == 0 for m in placeholders)

        completion_params = inspect.signature(LLAMA_CPP_STORAGE.llm.create_chat_completion).parameters
        final_params = {k: v for k, v in _parameters.items() if k in completion_params}

        # Check if the dialogue involves multimodal media (images/videos)
        has_media = len(all_images) > 0 or any(
            isinstance(msg.get("content"), list) and any(
                isinstance(item, dict) and item.get("type") in ("image_url", "image", "video_url", "video")
                for item in msg.get("content", [])
            )
            for msg in messages
        )

        # Model-backed draft sidecars (DFlash, DSpark, MTP) in llama.cpp only support text sequences.
        # When multimodal image tokens are evaluated, their 4D M-RoPE positions cannot be mapped to the draft context,
        # which causes llama_decode (code -1) in draft_context.
        # We automatically bypass model-backed draft engines for multimodal calls, keeping them active for text-only calls.
        spec_engine = getattr(LLAMA_CPP_STORAGE.llm, "speculative", None)
        has_draft_ctx = spec_engine is not None and getattr(spec_engine, "draft_context", None) is not None
        bypass_spec = has_media and has_draft_ctx
        if bypass_spec:
            print("[llama-cpp_vlm] Multimodal input detected: llama.cpp model-backed draft sidecars (DFlash/DSpark/MTP) are text-only; automatically bypassing draft model for this multimodal request.")
            LLAMA_CPP_STORAGE.llm.speculative = None

        try:
            if len(all_images) > 0:
                h = LLAMA_CPP_STORAGE.chat_handler
                h_path = getattr(h, "mmproj_path", getattr(h, "clip_model_path", None)) if h is not None else None
                if h_path is None:
                     raise ValueError("Image input detected, but the loaded model is not configured with a mmproj module.")
                    
                n_ctx = LLAMA_CPP_STORAGE.current_config.get("n_ctx", 8192) if LLAMA_CPP_STORAGE.current_config else 8192
                est_tokens = len(system_prompts) // 3 + len(prompt_text) // 3 + (len(all_images) * 1024)
                if est_tokens > n_ctx * 0.9:
                    print(f"[llama-cpp_vlm] Warning: Estimated prompt + multi-image tokens ({est_tokens}) close to or exceeding n_ctx ({n_ctx}). Consider increasing n_ctx in model loader.")

                frames = all_images
                if video_input and len(all_images) > max_frames:
                    indices = np.linspace(0, len(all_images) - 1, max_frames, dtype=int)
                    frames = [all_images[i] for i in indices]
                    
                if inference_mode == "one by one":
                    tmp_list = []
                    print(f"[llama-cpp_vlm] Start processing {len(frames)} images one by one")
                    
                    for i, image in enumerate(cqdm(frames)):
                        if mm.processing_interrupted():
                            raise mm.InterruptProcessingException()
                        if image.ndim == 4:
                            image = image.squeeze(0)
                        img_np = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
                        if img_np.ndim == 2:
                            img_np = np.stack([img_np] * 3, axis=-1)
                        data = image2base64(img_np)
                        
                        frame_user_content = [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
                        ]
                        frame_messages = messages + [{"role": "user", "content": frame_user_content}]
                        frame_seed = self.sanitize_seed(base_seed, offset=i)
                        try:
                            output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=frame_messages, seed=frame_seed, **final_params)
                        except Exception as e:
                            err_str = str(e)
                            if "context limit" in err_str.lower() or "eval_chunk_single" in err_str.lower() or "failed to find a memory slot" in err_str.lower() or "error code 1" in err_str.lower():
                                raise RuntimeError(
                                    f"Multimodal Context Limit Exceeded ({e}).\n\n"
                                    f"Your prompt and image generated more tokens than n_ctx={LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)}.\n"
                                    f"Qwen3.8 / Qwen3.6 / Qwen3.5 and M-RoPE models do not support context shifting in llama.cpp.\n"
                                    f"👉 Solution: Please increase 'n_ctx' in the Llama-cpp Model Loader node (e.g. from {LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)} to 16384 or 32768)."
                                ) from e
                            if "llama_decode failed" in err_str.lower() or "invalid input batch" in err_str.lower():
                                raise RuntimeError(
                                    f"Llama Decode Error ({e}).\n\n"
                                    "If using a draft sidecar model (DFlash/DSpark/MTP), note that llama.cpp draft sidecars are text-only."
                                ) from e
                            raise e
                        content = output['choices'][0]['message'].get('content', '') or ''
                        text = content.removeprefix(": ").lstrip()
                        out2.append(text)
                        if len(frames) > 1:
                            tmp_list.append(f"====== Image {i+1} ======")
                        tmp_list.append(text)
                        
                    out1 = "\n\n".join(tmp_list)
                else:
                    base64_frames = []
                    for img in frames:
                        if len(frames) > 1:
                            data = image2base64(scale_image(img, max_size))
                        else:
                            if img.ndim == 4:
                                img = img.squeeze(0)
                            img_np = np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
                            if img_np.ndim == 2:
                                img_np = np.stack([img_np] * 3, axis=-1)
                            data = image2base64(img_np)
                        base64_frames.append(f"data:image/jpeg;base64,{data}")

                    if placeholders:
                        last_idx = 0
                        for match in placeholders:
                            start, end = match.span()
                            num = int(match.group(1))
                            img_num = num if has_zero else num - 1
                            
                            text_chunk = prompt_text[last_idx:start]
                            if text_chunk:
                                user_content.append({"type": "text", "text": text_chunk})
                            
                            if 0 <= img_num < len(base64_frames):
                                user_content.append({"type": "text", "text": f"\n<Picture {num}>:\n"})
                                user_content.append({"type": "image_url", "image_url": {"url": base64_frames[img_num]}})
                            else:
                                user_content.append({"type": "text", "text": match.group(0)})
                            last_idx = end
                        
                        remaining_text = prompt_text[last_idx:]
                        if remaining_text:
                            user_content.append({"type": "text", "text": remaining_text})
                    else:
                        user_content.append({"type": "text", "text": prompt_text})
                        for idx, b64_url in enumerate(base64_frames):
                            if len(base64_frames) > 1:
                                tag_num = idx if has_zero else idx + 1
                                user_content.append({"type": "text", "text": f"\n<Picture {tag_num}>:\n"})
                            user_content.append({"type": "image_url", "image_url": {"url": b64_url}})

                    messages.append({"role": "user", "content": user_content})
                    try:
                        output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=active_seed, **final_params)
                    except Exception as e:
                        err_str = str(e)
                        if "context limit" in err_str.lower() or "eval_chunk_single" in err_str.lower() or "failed to find a memory slot" in err_str.lower() or "error code 1" in err_str.lower():
                            raise RuntimeError(
                                f"Multimodal Context Limit Exceeded ({e}).\n\n"
                                f"Your prompt and images generated more tokens than n_ctx={LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)}.\n"
                                f"Qwen3.8 / Qwen3.6 / Qwen3.5 and M-RoPE models do not support context shifting in llama.cpp.\n"
                                f"👉 Solution: Please increase 'n_ctx' in the Llama-cpp Model Loader node (e.g. from {LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)} to 16384 or 32768)."
                            ) from e
                        if "llama_decode failed" in err_str.lower() or "invalid input batch" in err_str.lower():
                            raise RuntimeError(
                                f"Llama Decode Error ({e}).\n\n"
                                "If using a draft sidecar model (DFlash/DSpark/MTP), note that llama.cpp draft sidecars are text-only."
                            ) from e
                        raise e
                    content = output['choices'][0]['message'].get('content', '') or ''
                    out1 = content.removeprefix(": ").lstrip()
                    out2 = [out1]
            else:
                user_content.append({"type": "text", "text": prompt_text})
                messages.append({"role": "user", "content": user_content})
                try:
                    output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=active_seed, **final_params)
                except Exception as e:
                    err_str = str(e)
                    if "context limit" in err_str.lower() or "eval_chunk_single" in err_str.lower() or "failed to find a memory slot" in err_str.lower() or "error code 1" in err_str.lower():
                        raise RuntimeError(
                            f"Multimodal Context Limit Exceeded ({e}).\n\n"
                            f"Your prompt and images generated more tokens than n_ctx={LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)}.\n"
                            f"Qwen3.8 / Qwen3.6 / Qwen3.5 and M-RoPE models do not support context shifting in llama.cpp.\n"
                            f"👉 Solution: Please increase 'n_ctx' in the Llama-cpp Model Loader node (e.g. from {LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)} to 16384 or 32768)."
                        ) from e
                    if "llama_decode failed" in err_str.lower() or "invalid input batch" in err_str.lower():
                        raise RuntimeError(
                            f"Llama Decode Error ({e}).\n\n"
                            "If using a draft sidecar model (DFlash/DSpark/MTP), note that llama.cpp draft sidecars are text-only."
                        ) from e
                    raise e
                content = output['choices'][0]['message'].get('content', '') or ''
                out1 = content.removeprefix(": ").lstrip()
                out2 = [out1]
        finally:
            if bypass_spec and spec_engine is not None:
                try:
                    spec_engine.clear()
                except Exception:
                    pass
                LLAMA_CPP_STORAGE.llm.speculative = spec_engine

        out1 = strip_think_block(out1)
            
        if save_states:
            print(f"[llama-cpp_vlm] Saving state id={uid}...")
            messages.append({"role": "assistant", "content": out1})
            clear_message = self.sanitize_messages(messages)
            LLAMA_CPP_STORAGE.messages[f"{uid}"] = clear_message
        else:
            if not LLAMA_CPP_STORAGE.messages.get(f"{uid}"):
                LLAMA_CPP_STORAGE.sys_prompts.pop(f"{uid}", None)
                
        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        else:
            if LLAMA_CPP_STORAGE.current_config and LLAMA_CPP_STORAGE.current_config.get("chat_handler") in [
                "Qwen3.5", "Qwen3.5-Thinking",
                "Qwen3.6", "Qwen3.6-Thinking",
                "Qwen3.8", "Qwen3.8-Thinking"
            ]:
                if LLAMA_CPP_STORAGE.llm is not None:
                    if hasattr(LLAMA_CPP_STORAGE.llm, "n_tokens"):
                        LLAMA_CPP_STORAGE.llm.n_tokens = 0
                    if hasattr(LLAMA_CPP_STORAGE.llm, "_ctx") and hasattr(LLAMA_CPP_STORAGE.llm._ctx, "memory_clear"):
                        try:
                            LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                        except Exception:
                            pass
                    if getattr(LLAMA_CPP_STORAGE.llm, "is_hybrid", False) and getattr(LLAMA_CPP_STORAGE.llm, "_hybrid_cache_mgr", None) is not None:
                        try:
                            LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()
                        except Exception:
                            pass
            
        del messages
        gc.collect()
        return (out1, out2, uid)

class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 4096, "min": -1, "max": 327680, "step": 1, "tooltip": "Max output tokens (-1 = uncapped up to context limit)."}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "stop": ("STRING", {"default": "", "multiline": False, "tooltip": "Comma-separated list of stop phrases to halt generation (e.g. '###, \\n\\n')."}),
                "reasoning_budget": ("INT", {"default": -1, "min": -1, "max": 32768, "step": 64, "tooltip": "Token budget for thinking models like Gemma4-Thinking / Qwen3.8-Thinking (-1 = no budget limit)."}),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Use a specific ID to save the conversation state.\n(-1 = use node's unique_id)"
                }),
            },
            "optional": {
                "dry_multiplier": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "DRY (Don't Repeat Yourself) repetition penalty multiplier (0.0 = disabled, e.g. 0.8). In llama-cpp-python 0.3.49 dry_penalty_last_n defaults to 64."}),
            }
        }
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    def process(self, **kwargs):
        stop_val = kwargs.get("stop", "")
        if isinstance(stop_val, str) and stop_val.strip():
            raw_stops = [s.strip() for s in stop_val.split(",") if s.strip()]
            parsed_stops = []
            for s in raw_stops:
                parsed = s.replace("\\n", "\n").replace("\\t", "\t")
                if parsed:
                    parsed_stops.append(parsed)
            if parsed_stops:
                kwargs["stop"] = parsed_stops
            else:
                kwargs.pop("stop", None)
        elif "stop" in kwargs:
            kwargs.pop("stop", None)

        if kwargs.get("reasoning_budget", -1) in (-1, 0):
            kwargs.pop("reasoning_budget", None)

        if kwargs.get("dry_multiplier", 0.0) == 0.0:
            kwargs.pop("dry_multiplier", None)

        return (kwargs,)
    
class llama_cpp_clean_states:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Clear the saved state for a specific ID (-1 = clear all)"
                }),
            },
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any, state_uid):
        print(f"[llama-cpp_vlm] Cleaning up saved states {state_uid}...")
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        return (any,)

class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,)}}
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any):
        print("[llama-cpp_vlm] Unloading llama model...")
        LLAMA_CPP_STORAGE.clean()
        return (any,)

class json_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple","Qwen3-VL", "Qwen2.5-VL"], {"default": "simple"}),
                "label": ("STRING", {
                    "default":"",
                    "multiline": False,
                    "tooltip": "Select only the BBoxes with specific labels."
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, json, mode, label, image=None):
        mode = mode[0] if isinstance(mode, list) and len(mode) > 0 else (mode if isinstance(mode, str) else "simple")
        label = label[0] if isinstance(label, list) and len(label) > 0 else (label if isinstance(label, str) else "")

        flat_images_list = []
        original_structure = []
    
        if image is not None:
            for img_batch in image:
                if img_batch.ndim == 3:
                    flat_images_list.append(img_batch.unsqueeze(0))
                    original_structure.append(1)
                else:
                    count = img_batch.shape[0]
                    original_structure.append(count)
                    for n in range(count):
                        flat_images_list.append(img_batch[n:n+1])
        
        total_images = len(flat_images_list)
        output_bboxes = []
        processed_flat_results = []
        
        for i, j in enumerate(json):
            bboxes = parse_json(j)
            
            if label != "":
                try:
                    bboxes = [item for item in bboxes if item["label"] == label]
                except Exception:
                    bboxes = [item for item in bboxes if item.get("text_content") == label]

            if total_images > 0:
                curr_idx = i if i < total_images else (total_images - 1)
                curr_img = flat_images_list[curr_idx]
                
                try:
                    res_img = draw_bbox(curr_img[0], bboxes, mode)
                    if res_img.ndim == 3:
                        res_img = res_img.unsqueeze(0)
                    elif res_img.ndim == 4 and res_img.shape[0] > 1:
                        res_img = res_img[0:1]
                        
                    processed_flat_results.append(res_img)
                except Exception as e:
                    print(f"Error drawing on image {curr_idx}: {e}")
                    processed_flat_results.append(curr_img)
                    
            if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
                if total_images == 0:
                    raise ValueError("Image required for Qwen mode")
                curr_idx = i if i < total_images else (total_images - 1)
                bbox = qwen3bbox(flat_images_list[curr_idx][0], bboxes)
            else:
                bbox = [tuple(item["bbox_2d"]) for item in bboxes if isinstance(item, dict) and "bbox_2d" in item]
                
            output_bboxes.append(bbox)
            
        restructured_images_list = []
        cursor = 0
        for count in original_structure:
            chunk = processed_flat_results[cursor : cursor + count]
            if chunk:
                restructured_images_list.append(torch.cat(chunk, dim=0))
            cursor += count
            
        return (output_bboxes, restructured_images_list)

class SEG:
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper=None):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region
        self.bbox = bbox
        self.label = label
        self.control_net_wrapper = control_net_wrapper
        
    def __repr__(self):
        return (f"SEG(cropped_image={self.cropped_image}, cropped_mask=shape{self.cropped_mask.shape}, confidence={self.confidence}, bbox={self.bbox}, label='{self.label}'), control_net_wrapper={self.control_net_wrapper}")
    
class bbox_to_segs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        
        seg_list = []
        image_for_cropping = image[0] 
        
        for bbox in bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            
            crop_region = [x1_exp, y1_exp, x2_exp, y2_exp]
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                print(f"Warning: Skipping bbox with invalid expanded size: {crop_region}")
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = local_x1 + (x2 - x1)
            local_y2 = local_y1 + (y2 - y1)
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter_2d(local_mask_np, sigma=feather)
                
            cropped_mask_np = local_mask_np
            cropped_img_padded = torch.zeros((crop_h, crop_w, 3), dtype=image.dtype, device=image.device)
            
            src_x_start = max(0, x1_exp)
            src_y_start = max(0, y1_exp)
            src_x_end = min(width, x2_exp)
            src_y_end = min(height, y2_exp)
            
            dst_x_start = src_x_start - x1_exp
            dst_y_start = src_y_start - y1_exp
            dst_x_end = src_x_end - x1_exp
            dst_y_end = src_y_end - y1_exp
            
            if src_x_end > src_x_start and src_y_end > src_y_start:
                source_crop = image_for_cropping[src_y_start:src_y_end, src_x_start:src_x_end, :]
                cropped_img_padded[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = source_crop
                
            cropped_image_tensor = cropped_img_padded.permute(2, 0, 1).unsqueeze(0)
            
            seg = SEG(
                cropped_image=cropped_image_tensor,
                cropped_mask=cropped_mask_np,
                confidence=np.array([0.9], dtype=np.float32),
                crop_region=crop_region,
                bbox=np.array(bbox, dtype=np.float32),
                label="bbox"
            )
            
            seg_list.append(seg)
            
        segs = (mask_shape, seg_list)
        
        return (segs,)
    
class bbox_to_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        masks = []
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        combined_full_mask = torch.zeros(mask_shape, dtype=torch.float32, device=image.device)
        
        for i, bbox in enumerate(bboxes):
            
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = local_x1 + (x2 - x1)
            local_y2 = local_y1 + (y2 - y1)
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter_2d(local_mask_np, sigma=feather)
                
            current_full_mask_np = np.zeros(mask_shape, dtype=np.float32)
            x1_c, y1_c = max(0, x1_exp), max(0, y1_exp)
            x2_c, y2_c = min(width, x2_exp), min(height, y2_exp)
            
            if x2_c > x1_c and y2_c > y1_c:
                dst_x1 = x1_c - x1_exp
                dst_y1 = y1_c - y1_exp
                dst_x2 = dst_x1 + (x2_c - x1_c)
                dst_y2 = dst_y1 + (y2_c - y1_c)
                current_full_mask_np[y1_c:y2_c, x1_c:x2_c] = local_mask_np[dst_y1:dst_y2, dst_x1:dst_x2]
                
            current_full_mask_tensor = torch.from_numpy(current_full_mask_np).to(image.device)
            combined_full_mask = torch.maximum(combined_full_mask, current_full_mask_tensor)
            
        masks.append(combined_full_mask.unsqueeze(0))
        return (torch.cat(masks, dim=0),)

class bboxes_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "bbox_index": ("INT", {
                    "default": 0,
                    "min": -998,
                    "max": 999,
                    "step": 1,
                    "tooltip": "BBox index in the image. Set to 999 to get all bboxes."
                }),
            }
        }
    
    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image_index, bbox_index):
        if not bboxes:
            return ([],)
        idx = min(image_index, len(bboxes) - 1)
        if bbox_index != 999:
            if idx < len(bboxes) and len(bboxes[idx]) > 0:
                b_idx = bbox_index if 0 <= bbox_index < len(bboxes[idx]) else (len(bboxes[idx]) - 1 if bbox_index < 0 else 0)
                return ([bboxes[idx][b_idx]],)
            return ([],)
        return (bboxes[idx],)

# from: https://github.com/crystian/ComfyUI-Crystools
# from: https://github.com/crystian/ComfyUI-Crystools
class parse_json_node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "key": ("STRING",),
                "default": ("STRING",),
            },
        }
    
    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, key=None, default=None):
        if isinstance(input, str):
            input = [input]
            
        res_any, res_string, res_int, res_float, res_boolean = [], [], [], [], []
        for json_str in input:
            if key is not None and key != "":
                cleaned = JSON_CODEBLOCK_PATTERN.sub("", json_str.strip())
                val = get_nested_value(cleaned, key, default)
            else:
                raise ValueError("Key cannot be empty!")
            
            res_any.append(val)
            try:
                res_string.append(str(val))
            except Exception:
                res_string.append(val)
            
            try:
                res_int.append(int(val))
            except Exception:
                res_int.append(val)
            
            try:
                res_float.append(float(val))
            except Exception:
                res_float.append(val)
            
            try:
                res_boolean.append(str(val).lower() == "true")
            except Exception:
                res_boolean.append(val)
                
        if len(res_any) == 1:
            return (res_any[0], res_string[0], res_int[0], res_float[0], res_boolean[0])
        return (res_any, res_string, res_int, res_float, res_boolean)

def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return default
        if isinstance(data, dict) and key in data:
            data = data[key]
        elif isinstance(data, list):
            try:
                idx = int(key)
                data = data[idx]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return data

class remove_code_block:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "label": ("STRING",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, label=None):
        if isinstance(input, str):
            input = [input]
        
        output = []
        pattern = re.compile(rf"^```(?:{re.escape(label)})?\s*|\s*```$", re.IGNORECASE) if label else JSON_CODEBLOCK_PATTERN
        for value in input:
            output.append(pattern.sub("", value.strip()))
        if len(output) == 1:
            return (output[0],)
        return (output,)

PRESET_LOOKUP = {
    "Qwen-Image [EN]": getattr(preset_mod, "QWEN_IMAGE_EN", ""),
    "Qwen-Image 2512 [EN]": getattr(preset_mod, "QWEN_IMAGE_2512_EN", ""),
    "Qwen-Image-Edit": getattr(preset_mod, "QWEN_IMAGE_EDIT", ""),
    "Qwen-Image-Edit 2509": getattr(preset_mod, "QWEN_IMAGE_EDIT_2509", ""),
    "Qwen-Image-Edit 2511": getattr(preset_mod, "QWEN_IMAGE_EDIT_2511", ""),
    "Z-Image Turbo": getattr(preset_mod, "ZIMAGE_TURBO", ""),
    "Flux.2 T2I": getattr(preset_mod, "FLUX2_T2I", ""),
    "Flux.2 I2I": getattr(preset_mod, "FLUX2_I2I", ""),
}

class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (list(PRESET_LOOKUP.keys()),)
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vlm"
    
    def main(self, preset):
        if preset in PRESET_LOOKUP:
            return (PRESET_LOOKUP[preset],)
        raise ValueError(f'Unknown preset: "{preset}"')
        
NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct_adv": llama_cpp_instruct_adv,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "llama_cpp_clean_states": llama_cpp_clean_states,
    "parse_json_node": parse_json_node,
    "json_to_bbox": json_to_bbox,
    "bbox_to_segs": bbox_to_segs,
    "bbox_to_mask": bbox_to_mask,
    "bboxes_to_bbox": bboxes_to_bbox,
    "remove_code_block": remove_code_block,
    "PromptEnhancerPreset": PromptEnhancerPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp Model Loader",
    "llama_cpp_instruct_adv": "Llama-cpp Instruct",
    "llama_cpp_parameters": "Llama-cpp Parameters",
    "llama_cpp_unload_model": "Llama-cpp Unload Model",
    "llama_cpp_clean_states": "Llama-cpp Clean States",
    "parse_json_node": "Parse JSON",
    "json_to_bbox": "JSON to BBoxes",
    "bbox_to_segs": "BBoxes to SEGS",
    "bbox_to_mask": "BBoxes to MASK",
    "bboxes_to_bbox": "BBoxes to BBox",
    "remove_code_block": "Unpack Code Block",
    "PromptEnhancerPreset": "Prompt Enhancer Preset",
}