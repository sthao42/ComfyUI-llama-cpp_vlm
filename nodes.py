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
from scipy.ndimage import gaussian_filter
from .support.cqdm import cqdm
from .support.gguf_layers import get_layer_count
from .support.prompt_enhancer_preset import *

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
except:
    Gemma3ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
    chat_handlers += ["Gemma4", "Gemma4-Thinking"]
except:
    Gemma4ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    chat_handlers += ["Qwen2.5-VL", "MinerU2.5-Pro"]
except:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    chat_handlers += ["Qwen3-VL", "Qwen3-VL-Thinking"]
except:
    Qwen3VLChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
    chat_handlers += ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking"]
except:
    Qwen35ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import (GLM46VChatHandler, LFM2VLChatHandler, GLM41VChatHandler)
    chat_handlers += ["GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "LFM2-VL"]
except:
    GLM46VChatHandler = None
    LFM2VLChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM25VLChatHandler
    chat_handlers += ["LFM2.5-VL"]
except:
    LFM25VLChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
    chat_handlers += ["Granite-Docling"]
except:
    GraniteDoclingChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import MiniCPMv45ChatHandler
    chat_handlers += ["MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"]
except:
    MiniCPMv45ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import MiniCPMv46ChatHandler
    chat_handlers += ["MiniCPM-v4.6", "MiniCPM-v4.6-Thinking"]
except:
    MiniCPMv46ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import PaddleOCRChatHandler
    chat_handlers += ["PaddleOCR-VL-1.5"]
except:
    PaddleOCRChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Qwen3ASRChatHandler
    chat_handlers += ["Qwen3-ASR"]
except:
    Qwen3ASRChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Step3VLChatHandler
    chat_handlers += ["Step3-VL"]
except:
    Step3VLChatHandler = None

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
    def clean_state(cls, id=-1):
        if id == -1:
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)
        
    @classmethod
    def clean(cls, all=False):
        try:
            cls.llm.close()
        except Exception:
            pass
            
        try:
            cls.chat_handler._exit_stack.close()
        except Exception:
            pass
        
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        if all:
            cls.clean_state()
        
        gc.collect()
        mm.soft_empty_cache()
    
    @classmethod
    def load_model(cls, config):
        def get_chat_handler(chat_handler):
            match chat_handler:
                case "Qwen3.5"|"Qwen3.5-Thinking"|"Qwen3.6"|"Qwen3.6-Thinking":
                    return Qwen35ChatHandler
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
                case "None":
                    return None
                case _:
                    raise ValueError(f'Unknown model type: "{chat_handler}"')
        
        cls.clean(all=True)
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
            if chat_handler in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                kwargs["force_reasoning"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
            elif any(name in chat_handler for name in ["MiniCPM-v4.5", "MiniCPM-v4.6", "GLM-4.6V", "GLM-4.1V", "Qwen3.5", "Qwen3.6", "Gemma4"]):
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
            
        if enable_mtp:
            try:
                from llama_cpp.llama_speculative import LlamaNGramMapDecoding
                llama_kwargs["draft_model"] = LlamaNGramMapDecoding()
                print("[llama-cpp_vlm] Multi-Token Prediction (MTP / Speculative Decoding) enabled using LlamaNGramMapDecoding.")
            except Exception as e:
                print(f"[llama-cpp_vlm] Warning: MTP (draft_model) failed to initialize: {e}")

        cls.llm = Llama(**llama_kwargs)

any_type = AnyType("*")

if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean(all=True)
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
    json_output = json_str.strip().removeprefix("```json").removesuffix("```")
    try:
        parsed = json.loads(json_output)
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed

def scale_image(image: torch.Tensor, max_size: int = 128):
    if image.ndim == 4:
        image = image.squeeze(0)
    img_np = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    img_pil = Image.fromarray(img_np)
    
    w, h = img_pil.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return np.array(img_resized)

def qwen3bbox(image, json):
    if image.ndim == 4:
        image = image.squeeze(0)
    img_np = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    img = Image.fromarray(img_np)
    bboxes = []
    for item in json:
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
    if image.ndim == 4:
        image = image.squeeze(0)
    img_np = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    
    for item in json:
        try:
            label = item["label"]
        except Exception:
            try:
                label = item["text_content"]
            except Exception:
                label = "bbox"
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
        text_size = draw.textbbox((x0, text_y), label)
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), label, fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

def strip_think_block(text: str) -> str:
    """Sanitize output text by stripping reasoning <think>...</think> blocks."""
    if not text:
        return ""
    cleaned = THINK_BLOCK_PATTERN.sub('', text)
    cleaned = THINK_BLOCK_UNCLOSED_PATTERN.sub('', cleaned)
    return cleaned.strip()

def collect_image_inputs(kwargs: dict) -> list:
    """Collect image and video frame inputs dynamically from kwargs (image_0..image_8 and video_0)."""
    all_images = []
    # Collect image sockets (image_0 to image_8 and images)
    for key in ["images"] + [f"image_{i}" for i in range(9)]:
        img_val = kwargs.get(key, None)
        if img_val is not None:
            if isinstance(img_val, list):
                all_images.extend(img_val)
            elif len(img_val.shape) == 4:
                for i in range(img_val.shape[0]):
                    all_images.append(img_val[i])
            else:
                all_images.append(img_val)
                
    # Collect video socket (video_0)
    vid_val = kwargs.get("video_0", None)
    if vid_val is not None:
        if isinstance(vid_val, list):
            all_images.extend(vid_val)
        elif len(vid_val.shape) == 4:
            for i in range(vid_val.shape[0]):
                all_images.append(vid_val[i])
        else:
            all_images.append(vid_val)
                
    return all_images

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = folder_paths.get_filename_list("LLM")
        model_list = [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"]+[f for f in all_llms if "mmproj" in f.lower()]
            
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
            "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
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
                "tooltip": "Enable Multi-Token Prediction (MTP / Speculative Decoding) using LlamaNGramMapDecoding to accelerate token generation."
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
            }
        }

    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"
    
    def loadmodel(self, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens, n_batch=2048, n_ubatch=512, enable_mtp=False, flash_attn=True, offload_kqv=True, kv_cache_type="f16", n_threads=0):
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
            "n_threads": n_threads
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
    
    def sanitize_messages(self, messages):
        clean_messages = messages.copy()
        for msg in clean_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        item["image_url"]["url"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"
        return clean_messages
    
    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, parameters=None, queue_handler=None, **kwargs):
        if seed is None or seed == -1:
            seed = random.randint(0, 0x7fffffff)
        else:
            seed = int(seed) & 0xFFFFFFFF

        if not LLAMA_CPP_STORAGE.llm:
            LLAMA_CPP_STORAGE.load_model(llama_model)
        
        if parameters is None:
            parameters = {}
        
        if _MTMD:
            parameters.pop("present_penalty", None)
            
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
            LLAMA_CPP_STORAGE.clean_state(id=uid)
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
                    try:
                        output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=frame_messages, seed=seed, **final_params)
                    except Exception as e:
                        err_str = str(e)
                        if "context limit" in err_str.lower() or "eval_chunk_single" in err_str.lower() or "failed to find a memory slot" in err_str.lower() or "error code 1" in err_str.lower():
                            raise RuntimeError(
                                f"Multimodal Context Limit Exceeded ({e}).\n\n"
                                f"Your prompt and image generated more tokens than n_ctx={LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)}.\n"
                                f"Qwen3.6 / Qwen3.5 and M-RoPE models do not support context shifting in llama.cpp.\n"
                                f"👉 Solution: Please increase 'n_ctx' in the Llama-cpp Model Loader node (e.g. from {LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)} to 16384 or 32768)."
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
                    output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **final_params)
                except Exception as e:
                    err_str = str(e)
                    if "context limit" in err_str.lower() or "eval_chunk_single" in err_str.lower() or "failed to find a memory slot" in err_str.lower() or "error code 1" in err_str.lower():
                        raise RuntimeError(
                            f"Multimodal Context Limit Exceeded ({e}).\n\n"
                            f"Your prompt and images generated more tokens than n_ctx={LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)}.\n"
                            f"Qwen3.6 / Qwen3.5 and M-RoPE models do not support context shifting in llama.cpp.\n"
                            f"👉 Solution: Please increase 'n_ctx' in the Llama-cpp Model Loader node (e.g. from {LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)} to 16384 or 32768)."
                        ) from e
                    raise e
                content = output['choices'][0]['message'].get('content', '') or ''
                out1 = content.removeprefix(": ").lstrip()
                out2 = [out1]
        else:
            user_content.append({"type": "text", "text": prompt_text})
            messages.append({"role": "user", "content": user_content})
            try:
                output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **final_params)
            except Exception as e:
                err_str = str(e)
                if "context limit" in err_str.lower() or "eval_chunk_single" in err_str.lower() or "failed to find a memory slot" in err_str.lower() or "error code 1" in err_str.lower():
                    raise RuntimeError(
                        f"Multimodal Context Limit Exceeded ({e}).\n\n"
                        f"Your prompt and images generated more tokens than n_ctx={LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)}.\n"
                        f"Qwen3.6 / Qwen3.5 and M-RoPE models do not support context shifting in llama.cpp.\n"
                        f"👉 Solution: Please increase 'n_ctx' in the Llama-cpp Model Loader node (e.g. from {LLAMA_CPP_STORAGE.current_config.get('n_ctx', 8192)} to 16384 or 32768)."
                    ) from e
                raise e
            content = output['choices'][0]['message'].get('content', '') or ''
            out1 = content.removeprefix(": ").lstrip()
            out2 = [out1]

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
            if LLAMA_CPP_STORAGE.current_config["chat_handler"] in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking"]:
                LLAMA_CPP_STORAGE.llm.n_tokens = 0
                LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                if LLAMA_CPP_STORAGE.llm.is_hybrid and LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr is not None:
                    LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()
            
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
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stop": ("STRING", {"default": "", "multiline": False, "tooltip": "Comma-separated list of stop phrases to halt generation (e.g. '###, \\n\\n')."}),
                "reasoning_budget": ("INT", {"default": -1, "min": -1, "max": 32768, "step": 64, "tooltip": "Token budget for thinking models like Gemma4-Thinking (-1 = no budget limit)."}),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Use a specific ID to save the conversation state.\n(-1 = use node's unique_id)"
                }),
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
        mode = mode[0]
        label = label[0]

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
                bbox = [tuple(item["bbox_2d"]) for item in bboxes]
                
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
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
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
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            current_full_mask_np = np.zeros(mask_shape, dtype=np.float32)
            x1_c, y1_c = max(0, x1_exp), max(0, y1_exp)
            x2_c, y2_c = min(width, x2_exp), min(height, y2_exp)
            
            if x2_c > x1_c and y2_c > y1_c:
                current_full_mask_np[y1_c:y2_c, x1_c:x2_c] = 1.0
                
            if feather > 0:
                current_full_mask_np = gaussian_filter(current_full_mask_np, sigma=feather)
                
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
        if bbox_index != 999:
            return ([bboxes[image_index][bbox_index]],)
        return (bboxes[image_index],)

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
            
        result = {}
        for i, json in enumerate(input):
            val = ""
            if key is not None and key != "":
                val = get_nested_value(json.strip().removeprefix("```json").removesuffix("```"), key, default)
            else:
                raise ValueError("Key cannot be empty!")
            
            result["any"][i] = val
            try:
                result["string"][i] = str(val)
            except Exception as e:
                result["string"][i] = val
            
            try:
                result["int"][i] = int(val)
            except Exception as e:
                result["int"][i] = val
            
            try:
                result["float"][i] = float(val)
            except Exception as e:
                result["float"][i] = val
            
            try:
                result["boolean"][i] = val.lower() == "true"
            except Exception as e:
                result["boolean"][i] = val
                
        if len(result["any"]) == 1:
            result["any"] = result["any"][0]
            result["string"] = result["string"][0]
            result["int"] = result["int"][0]
            result["float"] = result["float"][0]
            result["boolean"] = result["boolean"][0]
        
        return (result["any"], result["string"], result["int"], result["float"], result["boolean"])

def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
                data = json.loads(data)
        if isinstance(data, dict) and key in data:
            data = data[key]
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
    
    def process(self, input, label):
        if isinstance(input, str):
            input = [input]
        
        output = []
        for value in input:
            output.append(value.strip().removeprefix(f"```{label}").removesuffix("```"))
        if len(output) == 1:
            return (output[0],)
        return (output,)

class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (["Qwen-Image [EN]", "Qwen-Image [ZH]", "Qwen-Image 2512 [EN]", "Qwen-Image 2512 [ZH]", "Qwen-Image-Edit", "Qwen-Image-Edit 2509", "Qwen-Image-Edit 2511", "Z-Image Turbo", "Flux.2 T2I", "Flux.2 I2I", "Wan T2V [EN]", "Wan T2V [ZH]", "Wan I2V [EN]", "Wan I2V [ZH]", "Wan I2V Full-Auto [EN]", "Wan I2V Full-Auto [ZH]", "Wan FLF2V [EN]", "Wan FLF2V [ZH]"], )
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vlm"
    
    def main(self, preset):
        match preset:
            case "Qwen-Image [EN]":
                return (QWEN_IMAGE_EN,)
            case "Qwen-Image [ZH]":
                return (QWEN_IMAGE_ZH,)
            case "Qwen-Image 2512 [EN]":
                return (QWEN_IMAGE_2512_EN,)
            case "Qwen-Image 2512 [ZH]":
                return (QWEN_IMAGE_2512_ZH,)
            case "Qwen-Image-Edit":
                return (QWEN_IMAGE_EDIT,)
            case "Qwen-Image-Edit 2509":
                return (QWEN_IMAGE_EDIT_2509,)
            case "Qwen-Image-Edit 2511":
                return (QWEN_IMAGE_EDIT_2511,)
            case "Z-Image Turbo":
                return (ZIMAGE_TURBO,)
            case "Flux.2 T2I":
                return (FLUX2_T2I,)
            case "Flux.2 I2I":
                return (FLUX2_I2I,)
            case "Wan T2V [EN]":
                return (WAN_T2V_EN,)
            case "Wan T2V [ZH]":
                return (WAN_T2V_ZH,)
            case "Wan I2V [EN]":
                return (WAN_I2V_EN,)
            case "Wan I2V [ZH]":
                return (WAN_I2V_ZH,)
            case "Wan I2V Full-Auto [EN]":
                return (WAN_I2V_EMPTY_EN,)
            case "Wan I2V Full-Auto [ZH]":
                return (WAN_I2V_EMPTY_ZH,)
            case "Wan FLF2V [EN]":
                return (WAN_FLF2V_EN,)
            case "Wan FLF2V [ZH]":
                return (WAN_FLF2V_ZH,)
            case _:
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