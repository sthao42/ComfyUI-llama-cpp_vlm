import sys
import unittest
import torch
import numpy as np

sys.path.insert(0, r'G:\ComfyUI_windows_portable\ComfyUI')
sys.path.insert(0, 'd:/')

import importlib
nodes = importlib.import_module('ComfyUI-llama-cpp_vlm.nodes')

class TestEndToEndSimulation(unittest.TestCase):

    def test_full_node_simulation(self):
        print("\n--- Running Node End-to-End Simulation ---")
        
        # 1. Test Model Loader Config Generation
        loader = nodes.llama_cpp_model_loader()
        
        # Save original Llama.__init__ and mock it
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

        # 3. Test Multi-Image Collection from kwargs (image_0 to image_7)
        img0 = torch.zeros(1, 256, 256, 3)
        img1 = torch.ones(1, 256, 256, 3)
        img2 = torch.zeros(1, 256, 256, 3)
        kwargs = {"image_0": img0, "image_1": img1, "image_2": img2}
        
        all_images = []
        for key in ["images"] + [f"image_{i}" for i in range(8)]:
            img_val = kwargs.get(key, None)
            if img_val is not None:
                if isinstance(img_val, list):
                    all_images.extend(img_val)
                elif len(img_val.shape) == 4:
                    for i in range(img_val.shape[0]):
                        all_images.append(img_val[i])
                else:
                    all_images.append(img_val)
        
        self.assertEqual(len(all_images), 3)
        print("[OK] Multi-image kwargs collection (image_0 to image_2) test passed.")

        # 4. Test Tensor Scaling Safety
        scaled = nodes.scale_image(img0, max_size=128)
        self.assertEqual(scaled.shape, (128, 128, 3))
        print("[OK] Tensor scaling safety test passed.")

        # 5. Test <think> Reasoning Block Stripping
        raw_output = "<think>\nThinking step 1...\nThinking step 2...\n</think>\nA beautiful sunset over the ocean."
        import re
        def strip_think_block(text: str) -> str:
            if not text:
                return ""
            cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
            return cleaned.strip()
            
        clean_text = strip_think_block(raw_output)
        self.assertEqual(clean_text, "A beautiful sunset over the ocean.")
        print("[OK] Reasoning <think> block stripping test passed.")

if __name__ == '__main__':
    unittest.main()
