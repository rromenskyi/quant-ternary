"""Tests for the pipeline plumbing: pipeline_status (log + progress parsing,
shared by the shell pipelines and the dashboard), the dashboard's in-step
progress parser, and splice_common's key rules. No model weights needed;
the splice_common tests that build an mlx-lm module tree skip without mlx.

    python -m pytest poc/test_pipeline_tools.py -q
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_dashboard import inner_progress  # noqa: E402
from pipeline_status import calibration_state, parse_pipeline_log  # noqa: E402
from splice_common import is_dead_kv_shared, is_leftover, module_path_of  # noqa: E402

LOG = """\
PIPE 100 mlx START _pipeline mlx VARIANT=26b CORRECTED=/w/c OUT_DIR=/w/o
PIPE 100 mlx SKIP setup SETUP!=1
PIPE 101 mlx START download google/gemma-4-26B-A4B-it
PIPE 160 mlx DONE download /snap
PIPE 160 mlx START calibrate_text GPTQ text
PIPE 900 mlx DONE calibrate_text
PIPE 900 mlx START calibrate_vision GPTQ vision
PIPE 950 mlx FAIL calibrate_vision exit=3
PIPE 950 mlx FAIL _pipeline exit=3
PIPE 990 mlx START _pipeline mlx VARIANT=26b CORRECTED=/w/c OUT_DIR=/w/o
PIPE 991 mlx START calibrate_vision GPTQ vision
PIPE 500 gguf START _pipeline gguf
PIPE 501 gguf DONE _pipeline
"""


class TestPipelineLog(unittest.TestCase):
    def test_latest_run_only_and_statuses(self):
        p = parse_pipeline_log(LOG)
        mlx = p["mlx"]
        # The second START _pipeline resets the step list: only the rerun shows.
        self.assertEqual(mlx["order"], ["calibrate_vision"])
        self.assertEqual(mlx["status"], "running")
        self.assertEqual(mlx["steps"]["calibrate_vision"]["status"], "START")
        self.assertEqual(p["gguf"]["status"], "done")

    def test_durations_keep_start_time(self):
        p = parse_pipeline_log(LOG.split("PIPE 990")[0])
        st = p["mlx"]["steps"]["calibrate_text"]
        self.assertEqual((st["started"], st["ts"], st["status"]), (160, 900, "DONE"))
        self.assertEqual(p["mlx"]["status"], "failed")
        self.assertEqual(p["mlx"]["steps"]["calibrate_vision"]["detail"], "exit=3")


class TestCalibrationState(unittest.TestCase):
    def test_26b_text_needs_both_passes(self):
        config = {"text_config": {"num_hidden_layers": 3}, "vision_config": {"num_hidden_layers": 2}}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "gptq_progress_text.json").write_text(json.dumps({"done_layers": [0, 1, 2]}))
            (Path(d) / "gptq_progress_text_moe.json").write_text(json.dumps({"done_layers": [0, 1]}))
            (Path(d) / "gptq_progress_vision.json").write_text(json.dumps({"done_layers": [0, 1]}))
            st = calibration_state(config, Path(d), "26b")
        self.assertFalse(st["text"]["complete"])  # MoE pass still missing layer 2
        self.assertTrue(st["vision"]["complete"])

    def test_e4b_has_audio_and_missing_file_is_empty(self):
        config = {"text_config": {"num_hidden_layers": 2}, "vision_config": {"num_hidden_layers": 1},
                  "audio_config": {"num_hidden_layers": 1}}
        with tempfile.TemporaryDirectory() as d:
            st = calibration_state(config, Path(d), "e4b")
        self.assertEqual(set(st), {"text", "vision", "audio"})
        self.assertEqual(st["audio"]["passes"]["audio"], [])


class TestInnerProgress(unittest.TestCase):
    def test_calibration_and_quantize_lines(self):
        text = "--- text batch 1/5: layers [0, 1] ---\n    [2/6]\n--- text_moe batch 3/30: layers [2] ---\n    [4/6]\n"
        got = {x["label"]: (x["n"], x["of"]) for x in inner_progress(text)}
        self.assertEqual(got["text_moe batch"], (3, 30))
        self.assertEqual(got["calibration example"], (4, 6))
        got = {x["label"]: (x["n"], x["of"]) for x in inner_progress("[ 12/ 700] blk.0.attn_q.weight - [2816, 4096]")}
        self.assertEqual(got["llama-quantize tensor"], (12, 700))


class TestSpliceCommon(unittest.TestCase):
    def test_module_path(self):
        self.assertEqual(module_path_of("model.language_model.layers.0.mlp.down_proj.weight"),
                         "language_model.model.layers.0.mlp.down_proj")
        self.assertEqual(module_path_of("model.embed_vision.embedding_projection.weight"),
                         "embed_vision.embedding_projection")

    def test_dead_kv_shared(self):
        cfg = {"text_config": {"num_hidden_layers": 42, "num_kv_shared_layers": 18}}
        self.assertTrue(is_dead_kv_shared("model.language_model.layers.24.self_attn.k_proj.weight", cfg))
        self.assertTrue(is_dead_kv_shared("model.language_model.layers.41.self_attn.k_norm.weight", cfg))
        self.assertFalse(is_dead_kv_shared("model.language_model.layers.23.self_attn.k_proj.weight", cfg))
        self.assertFalse(is_dead_kv_shared("model.language_model.layers.30.self_attn.q_proj.weight", cfg))
        self.assertFalse(is_dead_kv_shared("model.language_model.layers.30.self_attn.k_proj.weight",
                                           {"text_config": {"num_hidden_layers": 42, "num_kv_shared_layers": 0}}))

    def test_leftover_rules(self):
        q = {"audio_tower.output_proj", "vision_tower.patch_embedder.input_proj", "language_model.model.layers.0.router.proj"}
        keep = [re.compile(r"patch_embedder\.input_proj"), re.compile(r"router\.proj")]
        big = [1536, 1024]
        self.assertTrue(is_leftover("model.audio_tower.output_proj.weight", big, "BF16", q, 100_000, keep))
        self.assertFalse(is_leftover("model.vision_tower.patch_embedder.input_proj.weight", big, "BF16", q, 100_000, keep))
        self.assertFalse(is_leftover("model.language_model.layers.0.router.proj.weight", big, "BF16", q, 100_000, keep))
        self.assertFalse(is_leftover("model.audio_tower.output_proj.weight", [10, 10], "BF16", q, 100_000, keep))
        self.assertFalse(is_leftover("model.audio_tower.conv.weight", big, "BF16", q, 100_000, keep))  # not a quantizable module
        self.assertFalse(is_leftover("model.audio_tower.output_proj.weight", big, "U32", q, 100_000, keep))


try:
    import mlx_lm  # noqa: F401
    HAVE_MLX_LM = True
except ImportError:
    HAVE_MLX_LM = False


@unittest.skipUnless(HAVE_MLX_LM, "needs mlx-lm (ipsupport-llc fork)")
class TestQuantizableModules(unittest.TestCase):
    def test_gemma4_text_tree(self):
        from splice_common import quantizable_module_paths

        cfg = {"model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
               "num_attention_heads": 2, "head_dim": 32, "global_head_dim": 64, "vocab_size": 128,
               "num_key_value_heads": 1, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
               "layer_types": ["sliding_attention", "full_attention"]}
        q = quantizable_module_paths(cfg)
        self.assertIn("model.layers.0.self_attn.q_proj", q)
        self.assertIn("model.embed_tokens", q)
        self.assertNotIn("model.layers.0.input_layernorm", q)


if __name__ == "__main__":
    unittest.main()
