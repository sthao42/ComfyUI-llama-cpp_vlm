import os
import sys
import types
import unittest
import torch
import numpy as np

# Dynamic path resolution: add project root directory to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Mock external ComfyUI and llama_cpp dependencies if running outside ComfyUI environment
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
        def __init__(self, model_path=None, chat_handler=None, n_gpu_layers=None, n_ctx=None, n_batch=None, n_ubatch=None, speculative=None, draft_model=None, flash_attn=None, offload_kqv=None, type_k=None, type_v=None, n_threads=None, verbose=False, **kwargs): pass
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
        "Qwen25VLChatHandler", "Qwen3VLChatHandler", "Qwen35ChatHandler", "Qwen38ChatHandler",
        "GLM46VChatHandler", "LFM2VLChatHandler", "GLM41VChatHandler",
        "LFM25VLChatHandler", "GraniteDoclingChatHandler", "MiniCPMv45ChatHandler",
        "MiniCPMv46ChatHandler", "PaddleOCRChatHandler", "Qwen3ASRChatHandler", "Step3VLChatHandler"
    ]:
        setattr(chat_fmt, h_name, DummyHandler)
    llama_cpp.llama_chat_format = chat_fmt
    llama_spec = types.ModuleType("llama_cpp.llama_speculative")
    import enum
    class SpeculativeType(enum.IntEnum):
        NONE = 0
        DRAFT_SIMPLE = 1
        DRAFT_EAGLE3 = 2
        DRAFT_MTP = 3
        DRAFT_DFLASH = 4
        DRAFT_DSPARK = 5
        NGRAM_SIMPLE = 6
        NGRAM_MAP_K = 7
        NGRAM_MAP_K4V = 8
        NGRAM_MOD = 9
        NGRAM_CACHE = 10
    class SpecConfig:
        def __init__(self, spec_type=SpeculativeType.NONE, **kwargs):
            self.spec_type = spec_type
            for k, v in kwargs.items():
                setattr(self, k, v)
    class LlamaNGramMapDecoding:
        def __init__(self, **kwargs): pass
    llama_spec.SpeculativeType = SpeculativeType
    llama_spec.SpecConfig = SpecConfig
    llama_spec.LlamaNGramMapDecoding = LlamaNGramMapDecoding
    llama_cpp.llama_speculative = llama_spec

    sys.modules["llama_cpp"] = llama_cpp
    sys.modules["llama_cpp.llama_chat_format"] = chat_fmt
    sys.modules["llama_cpp.llama_speculative"] = llama_spec

import nodes

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
        raw = {
            "max_tokens": 2048,
            "stop": "###, \\n\\n, User:",
            "reasoning_budget": -1
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
        tensor_4d = torch.zeros(1, 256, 256, 3)
        res_4d = nodes.scale_image(tensor_4d, max_size=128)
        self.assertEqual(res_4d.shape, (128, 128, 3))

        tensor_3d = torch.zeros(256, 256, 3)
        res_3d = nodes.scale_image(tensor_3d, max_size=128)
        self.assertEqual(res_3d.shape, (128, 128, 3))

        tensor_2d = torch.zeros(256, 256)
        res_2d = nodes.scale_image(tensor_2d, max_size=128)
        self.assertEqual(res_2d.shape, (128, 128, 3))

    def test_think_block_stripping(self):
        sample_closed = "<think>\nInternal reasoning steps...\n</think>\nA red car on a sunny street."
        self.assertEqual(nodes.strip_think_block(sample_closed), "A red car on a sunny street.")

        sample_unclosed = "<think>\nGeneration cut off during thinking..."
        self.assertEqual(nodes.strip_think_block(sample_unclosed), "")

    def test_seed_sanitization_and_is_changed(self):
        inst = nodes.llama_cpp_instruct_adv()
        import math
        self.assertTrue(math.isnan(inst.IS_CHANGED(None, "", "", "", "batch", 1, 256, -1, False, False, None)))
        self.assertEqual(inst.IS_CHANGED(None, "", "", "", "batch", 1, 256, 42, False, True, None), "42_True")
        self.assertEqual(inst.sanitize_seed(0xFFFFFFFF), 0xFFFFFFFF - 1)
        self.assertEqual(inst.sanitize_seed(0xFFFFFFFFFFFFFFFF), 0xFFFFFFFF - 1)
        self.assertEqual(inst.sanitize_seed(50, offset=5), 55)

    def test_gaussian_filter_2d_native(self):
        arr = np.zeros((20, 20), dtype=np.float32)
        arr[10, 10] = 1.0
        blurred = nodes.gaussian_filter_2d(arr, sigma=2.0)
        self.assertEqual(blurred.shape, (20, 20))
        self.assertTrue(blurred[10, 10] < 1.0)
        self.assertTrue(blurred[10, 10] > 0.0)

    def test_parse_json_robustness(self):
        json_raw = "```json\n{\"bbox_2d\": [10, 20, 30, 40], \"label\": \"dog\"}\n```"
        parsed = nodes.parse_json(json_raw)
        self.assertEqual(parsed["label"], "dog")

    def test_qwen38_support(self):
        for model_name in ["Qwen3.8", "Qwen3.8-Thinking"]:
            self.assertIn(model_name, nodes.chat_handlers)
        self.assertNotIn("Qwen3.8-27B", nodes.chat_handlers)
        self.assertNotIn("Qwen3.8-27B-Thinking", nodes.chat_handlers)

    def test_flatten_image_tensors_and_collection(self):
        t1 = torch.zeros(2, 64, 64, 3)
        t2 = torch.zeros(1, 64, 64, 3)
        kwargs = {"image_0": t1, "video_0": [t2, t1]}
        collected = nodes.collect_image_inputs(kwargs)
        self.assertEqual(len(collected), 5)

    def test_get_nested_value_array_support(self):
        data = {"items": [{"name": "first"}, {"name": "second"}]}
        self.assertEqual(nodes.get_nested_value(data, "items.1.name"), "second")
        self.assertEqual(nodes.get_nested_value(data, "items.5.name", default="none"), "none")

    def test_bboxes_boundary_safety(self):
        node = nodes.bboxes_to_bbox()
        self.assertEqual(node.process([], 0, 0), ([],))
        self.assertEqual(node.process([[(10, 20, 30, 40)]], 5, 0), ([(10, 20, 30, 40)],))
        self.assertEqual(node.process([[(10, 20, 30, 40)]], 0, 999), ([(10, 20, 30, 40)],))

    def test_speculative_config_integration(self):
        loader = nodes.llama_cpp_model_loader()
        captured_kwargs = {}
        def mock_init(self, model_path=None, chat_handler=None, n_gpu_layers=None, n_ctx=None, n_batch=None, n_ubatch=None, speculative=None, draft_model=None, **kwargs):
            captured_kwargs.update(kwargs)
            if speculative is not None:
                captured_kwargs["speculative"] = speculative
            if draft_model is not None:
                captured_kwargs["draft_model"] = draft_model
        orig_init = nodes.Llama.__init__
        try:
            nodes.Llama.__init__ = mock_init
            loader.loadmodel(
                model="fake_model.gguf",
                mmproj="None",
                chat_handler="None",
                n_ctx=4096,
                vram_limit=-1,
                image_min_tokens=0,
                image_max_tokens=0,
                enable_mtp=True
            )
        finally:
            nodes.Llama.__init__ = orig_init

        from llama_cpp.llama_speculative import SpecConfig, SpeculativeType
        self.assertIn("speculative", captured_kwargs)
        spec = captured_kwargs["speculative"]
        self.assertIsInstance(spec, SpecConfig)
        self.assertEqual(spec.spec_type, SpeculativeType.NGRAM_MAP_K)


if __name__ == '__main__':
    unittest.main()
