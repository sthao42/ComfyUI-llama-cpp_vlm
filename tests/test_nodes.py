import sys
import unittest
import torch
import numpy as np

# Ensure ComfyUI and node modules are in sys.path
sys.path.insert(0, r'G:\ComfyUI_windows_portable\ComfyUI')
sys.path.insert(0, 'd:/')

import importlib
nodes = importlib.import_module('ComfyUI-llama-cpp_vlm.nodes')

class TestComfyUILlamaCppVLM(unittest.TestCase):

    def test_model_loader_input_types(self):
        inputs = nodes.llama_cpp_model_loader.INPUT_TYPES()
        required_keys = list(inputs['required'].keys())
        expected = ['model', 'mmproj', 'chat_handler', 'n_ctx', 'vram_limit', 'image_min_tokens', 
                    'image_max_tokens', 'n_batch', 'n_ubatch', 'enable_mtp', 'flash_attn', 
                    'offload_kqv', 'kv_cache_type', 'n_threads']
        for key in expected:
            self.assertIn(key, required_keys, f"Missing {key} in model loader required inputs")

    def test_parameters_node_processing(self):
        params_node = nodes.llama_cpp_parameters()
        # Test stop word split & advanced sampler filtering
        raw = {
            "max_tokens": 2048,
            "stop": "###, \\n\\n, User:",
            "reasoning_budget": -1  # Should be popped
        }
        res = params_node.process(**raw)[0]
        self.assertEqual(res["stop"], ["###", "\n\n", "User:"])
        self.assertNotIn("reasoning_budget", res)

    def test_instruct_adv_optional_inputs(self):
        inputs = nodes.llama_cpp_instruct_adv.INPUT_TYPES()
        optional_keys = list(inputs['optional'].keys())
        for i in range(9):
            self.assertIn(f'image_{i}', optional_keys)
        self.assertIn('video_0', optional_keys)
        self.assertNotIn('video_1', optional_keys)

    def test_scale_image_safety(self):
        # Test 4D tensor [1, 256, 256, 3]
        tensor_4d = torch.zeros(1, 256, 256, 3)
        res_4d = nodes.scale_image(tensor_4d, max_size=128)
        self.assertEqual(res_4d.shape, (128, 128, 3))

        # Test 3D tensor [256, 256, 3]
        tensor_3d = torch.zeros(256, 256, 3)
        res_3d = nodes.scale_image(tensor_3d, max_size=128)
        self.assertEqual(res_3d.shape, (128, 128, 3))

        # Test 2D / 1-channel tensor [256, 256]
        tensor_2d = torch.zeros(256, 256)
        res_2d = nodes.scale_image(tensor_2d, max_size=128)
        self.assertEqual(res_2d.shape, (128, 128, 3))

    def test_think_block_stripping(self):
        # We simulate the strip_think_block logic
        import re
        def strip_think_block(text: str) -> str:
            if not text:
                return ""
            cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
            return cleaned.strip()

        sample_closed = "<think>\nInternal reasoning steps...\n</think>\nA red car on a sunny street."
        self.assertEqual(strip_think_block(sample_closed), "A red car on a sunny street.")

        sample_unclosed = "<think>\nGeneration cut off during thinking..."
        self.assertEqual(strip_think_block(sample_unclosed), "")

if __name__ == '__main__':
    unittest.main()
