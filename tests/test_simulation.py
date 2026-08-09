import os
import sys
import types
import unittest
import torch
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if "folder_paths" not in sys.modules:
    fp = types.ModuleType("folder_paths")
    fp.models_dir = os.path.join(REPO_ROOT, "models")
    fp.folder_names_and_paths = {}
    fp.get_filename_list = lambda x: []
    sys.modules["folder_paths"] = fp

if "comfy" not in sys.modules:
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.processing_interrupted = lambda: False
    mm.InterruptProcessingException = Exception
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda: None
    comfy.model_management = mm

    utils = types.ModuleType("comfy.utils")
    class DummyProgressBar:
        def __init__(self, total=None): self.total = total
        def update(self, n=1): pass
    utils.ProgressBar = DummyProgressBar
    comfy.utils = utils

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.utils"] = utils

if "llama_cpp" not in sys.modules:
    llama_cpp = types.ModuleType("llama_cpp")
    class DummyLlama:
        def __init__(self, **kwargs): pass
        def close(self): pass
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": "Mock completion response"}}]}
    llama_cpp.Llama = DummyLlama

    chat_fmt = types.ModuleType("llama_cpp.llama_chat_format")
    class DummyHandler:
        def __init__(self, **kwargs): pass
    for h_name in [
        "Llava15ChatHandler", "Llava16ChatHandler", "MoondreamChatHandler",
        "NanoLlavaChatHandler", "Llama3VisionAlphaChatHandler", "MiniCPMv26ChatHandler",
        "MTMDChatHandler", "Gemma3ChatHandler", "Gemma4ChatHandler",
        "Qwen25VLChatHandler", "Qwen3VLChatHandler", "Qwen35ChatHandler",
        "GLM46VChatHandler", "LFM2VLChatHandler", "GLM41VChatHandler",
        "LFM25VLChatHandler", "GraniteDoclingChatHandler", "MiniCPMv45ChatHandler",
        "MiniCPMv46ChatHandler", "PaddleOCRChatHandler", "Qwen3ASRChatHandler", "Step3VLChatHandler"
    ]:
        setattr(chat_fmt, h_name, DummyHandler)
    llama_cpp.llama_chat_format = chat_fmt
    sys.modules["llama_cpp"] = llama_cpp
    sys.modules["llama_cpp.llama_chat_format"] = chat_fmt

import nodes

class TestEndToEndSimulation(unittest.TestCase):

    def test_full_node_simulation(self):
        print("\n--- Running Node End-to-End Simulation ---")
        
        # 1. Test Model Loader Config Generation
        loader = nodes.llama_cpp_model_loader()
        
        orig_init = nodes.Llama.__init__
        try:
            nodes.Llama.__init__ = lambda self, **kwargs: None
            cfg = loader.loadmodel(
                model="fake_model.gguf",
                mmproj="None",
                chat_handler="None",
                n_ctx=8192,
                vram_limit=-1,
                image_min_tokens=0,
                image_max_tokens=0,
                n_batch=2048,
                n_ubatch=512,
                enable_mtp=True,
                flash_attn=True,
                offload_kqv=True,
                kv_cache_type="q8_0",
                n_threads=8
            )[0]
        finally:
            nodes.Llama.__init__ = orig_init
        
        self.assertEqual(cfg["flash_attn"], True)
        self.assertEqual(cfg["offload_kqv"], True)
        self.assertEqual(cfg["kv_cache_type"], "q8_0")
        self.assertEqual(cfg["n_threads"], 8)
        print("[OK] Model loader configuration test passed.")

        # 2. Test Parameters Node Processing
        param_node = nodes.llama_cpp_parameters()
        params = param_node.process(
            max_tokens=4096,
            temperature=0.7,
            stop="###, \\n\\n, User:",
            reasoning_budget=-1
        )[0]
        
        self.assertEqual(params["stop"], ["###", "\n\n", "User:"])
        self.assertNotIn("reasoning_budget", params)
        print("[OK] Parameter node processing test passed.")

        # 3. Test Multi-Image Collection from kwargs
        img0 = torch.zeros(1, 256, 256, 3)
        img1 = torch.ones(1, 256, 256, 3)
        img2 = torch.zeros(1, 256, 256, 3)
        kwargs = {"image_0": img0, "image_1": img1, "image_2": img2}
        
        all_images = nodes.collect_image_inputs(kwargs)
        self.assertEqual(len(all_images), 3)
        print("[OK] Multi-image kwargs collection test passed.")

        # 4. Test Tensor Scaling Safety
        scaled = nodes.scale_image(img0, max_size=128)
        self.assertEqual(scaled.shape, (128, 128, 3))
        print("[OK] Tensor scaling safety test passed.")

        # 5. Test <think> Reasoning Block Stripping
        raw_output = "<think>\nThinking step 1...\nThinking step 2...\n</think>\nA beautiful sunset over the ocean."
        clean_text = nodes.strip_think_block(raw_output)
        self.assertEqual(clean_text, "A beautiful sunset over the ocean.")
        print("[OK] Reasoning <think> block stripping test passed.")

        # 6. Test Seed Sanitization, IS_CHANGED & Controls
        inst = nodes.llama_cpp_instruct_adv()
        
        changed_random = inst.IS_CHANGED(None, "", "", "", "batch", 1, 256, -1, False, False, None)
        import math
        self.assertTrue(math.isnan(changed_random))
        
        changed_fixed = inst.IS_CHANGED(None, "", "", "", "batch", 1, 256, 12345, False, False, None)
        self.assertEqual(changed_fixed, "12345_False")

        sanitized_rand = inst.sanitize_seed(-1)
        self.assertTrue(0 <= sanitized_rand <= 0x7fffffff)

        sanitized_default = inst.sanitize_seed(0xFFFFFFFF)
        self.assertEqual(sanitized_default, 0xFFFFFFFF - 1)

        seed_64bit = 18446744073709551615
        sanitized_64bit = inst.sanitize_seed(seed_64bit)
        self.assertEqual(sanitized_64bit, 0xFFFFFFFF - 1)

        sanitized_frame = inst.sanitize_seed(100, offset=3)
        self.assertEqual(sanitized_frame, 103)
        print("[OK] Seed control sanitization & IS_CHANGED caching tests passed.")

if __name__ == '__main__':
    unittest.main()
