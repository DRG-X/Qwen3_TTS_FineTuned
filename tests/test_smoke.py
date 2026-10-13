"""CPU smoke tests. No model weights, no GPU, no network needed (except the optional real-adapter test)."""
import json
import os
import re
import shutil
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- config
def test_original_config_covers_every_key_the_scripts_read():
    cfg = json.load(open(os.path.join(ROOT, "configs", "train_hindi_female.original.json"), encoding="utf-8"))
    used = set()
    for fn in ("htts_common.py", "htts_prepare_data.py", "htts_train.py", "htts_export.py"):
        src = open(os.path.join(ROOT, "training", fn), encoding="utf-8").read()
        used |= set(re.findall(r'cfg\["([a-z_0-9]+)"\]', src))
    missing = sorted(k for k in used if k not in cfg)
    assert not missing, f"scripts read config keys absent from the config: {missing}"


def test_original_config_pins_documented_values():
    cfg = json.load(open(os.path.join(ROOT, "configs", "train_hindi_female.original.json"), encoding="utf-8"))
    assert cfg["base_model"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert cfg["tokenizer_model"] == "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    assert cfg["hf_dataset"] == "SPRINGLab/IndicTTS-Hindi"
    assert (cfg["lora_r"], cfg["cp_lora_r"], cfg["lr"], cfg["num_epochs"]) == (32, 8, 1e-4, 6)
    assert cfg["speaker_name"] == "hindi_female"


# ----------------------------------------------------------------------------- inference wrapper
class _FakeTTS:
    def __init__(self):
        self.calls = []

    def generate_custom_voice(self, **kw):
        self.calls.append(kw)
        return [[0.0, 0.1]], 24000


def test_wrapper_forces_auto_language_non_streaming_and_speaker():
    from qwen3_tts_hindi import HindiTTS, default_max_new_tokens

    fake = _FakeTTS()
    _wav, sr = HindiTTS(fake).synthesize("नमस्ते, आप कैसे हैं?")
    kw = fake.calls[0]
    assert kw["language"] == "auto" and kw["non_streaming_mode"] is True and kw["speaker"] == "hindi_female"
    assert kw["max_new_tokens"] == default_max_new_tokens("नमस्ते, आप कैसे हैं?") and sr == 24000


def test_wrapper_input_validation():
    from qwen3_tts_hindi import HindiTTS, has_devanagari

    t = HindiTTS(_FakeTTS())
    with pytest.raises(ValueError):
        t.synthesize("   ")
    with pytest.warns(UserWarning):
        t.synthesize("hello world")
    assert has_devanagari("हिन्दी") and not has_devanagari("hindi")


def test_max_new_tokens_formula_matches_export_script():
    from qwen3_tts_hindi import default_max_new_tokens

    src = open(os.path.join(ROOT, "training", "htts_export.py"), encoding="utf-8").read()
    assert "return int(len(text) * 2.5) + 60" in src
    assert default_max_new_tokens("x" * 100) == 310


# ----------------------------------------------------------------------------- torch-dependent
torch = pytest.importorskip("torch")
import htts_common as H
from torch import nn


def test_collator_layout_matches_documented_layout():
    c = types.SimpleNamespace(tts_pad_token_id=190, tts_bos_token_id=191, tts_eos_token_id=192)
    t = types.SimpleNamespace(codec_nothink_id=102, codec_think_bos_id=103, codec_think_eos_id=104, codec_pad_id=105,
                              codec_bos_id=106, codec_eos_token_id=100, num_code_groups=4)
    hf = types.SimpleNamespace(**vars(c), talker_config=t)
    ti = torch.tensor([7, 8, 9, 10, 11, 12, 13])          # 3 role tokens + 4 text tokens, L=7
    ac = torch.arange(12).view(3, 4) % 50                  # N=3 frames
    b = H.TTSCollator(hf)([{"text_ids": ti, "audio_codes": ac}])
    L, N = 7, 3
    s, end = 8 + L - 1, 8 + L + N
    assert b["text_in"][0, :3].tolist() == [7, 8, 9]
    assert b["text_in"][0, 3:7].tolist() == [190] * 4 and b["text_in"][0, 7].item() == 191
    assert b["text_in"][0, 8:8 + L - 3].tolist() == [10, 11, 12, 13]
    assert b["text_in"][0, 8 + L - 3].item() == 192
    assert b["text_in"][0, 8 + L - 2:end].tolist() == [190] * (end - (8 + L - 2))
    assert b["codec_in"][0, 3:8].tolist() == [102, 103, 104, 0, 105]
    assert b["codec_in"][0, 8:8 + L - 2].tolist() == [105] * (L - 2)
    assert b["codec_in"][0, 8 + L - 2].item() == 106
    assert b["codec_in"][0, s:s + N].tolist() == ac[:, 0].tolist() and b["codec_in"][0, s + N].item() == 100
    assert b["labels"][0, s:s + N].tolist() == ac[:, 0].tolist() and b["labels"][0, s + N].item() == 100
    assert (b["labels"][0, :s] == -100).all() and (b["labels"][0, s + N + 1:] == -100).all()
    assert not b["codec_emb_mask"][0, 6] and b["codec_emb_mask"][0, 3] and b["codec_emb_mask"][0, end - 1]
    assert b["codec_mask"][0, s:s + N].all() and b["codec_mask"][0].sum() == N
    assert b["attn"][0, :end].all() and b["attn"][0].sum() == end


class _Block(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.self_attn = nn.ModuleDict({k: nn.Linear(d, d, bias=False) for k in ("q_proj", "k_proj", "v_proj", "o_proj")})
        self.mlp = nn.ModuleDict({k: nn.Linear(d, d, bias=False) for k in ("gate_proj", "up_proj", "down_proj")})


def _tiny_core(d=8):
    core = nn.Module()
    core.talker = nn.Module()
    core.talker.model = nn.Module()
    core.talker.model.layers = nn.ModuleList([_Block(d)])
    core.talker.text_projection = nn.Module()
    core.talker.text_projection.linear_fc1 = nn.Linear(d, d)
    core.talker.text_projection.linear_fc2 = nn.Linear(d, d)
    core.talker.code_predictor = nn.Module()
    core.talker.code_predictor.model = nn.Module()
    core.talker.code_predictor.model.layers = nn.ModuleList([_Block(d)])
    return core


GROUPS = {"talker": (4, 4), "text_projection": (4, 4), "code_predictor": (2, 2)}


def test_lora_injection_targets_and_freezing():
    core = _tiny_core()
    counts = H.inject_lora(core, GROUPS, dropout=0.0)
    assert counts == {"talker": 7, "text_projection": 2, "code_predictor": 7}
    trainable = [n for n, p in core.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in n for n in trainable)


def test_lora_merge_equals_adapter_forward_fp32():
    core = _tiny_core()
    H.inject_lora(core, GROUPS, dropout=0.0)
    for n, p in core.named_parameters():
        if n.endswith("lora_B"):
            nn.init.normal_(p, std=0.05)
    x = torch.randn(5, 8)
    mods = {n: m for n, m in core.named_modules() if isinstance(m, H.LoRALinear)}
    before = {n: m(x).detach().clone() for n, m in mods.items()}
    assert H.merge_lora(core) == len(mods) == 16
    for n in mods:
        merged = core.get_submodule(n)
        assert isinstance(merged, nn.Linear)
        assert torch.allclose(merged(x), before[n], atol=1e-5)


def test_lora_merge_into_bf16_error_is_small_but_nonzero():
    """Informational bound: the notebook merges into bf16 weights; measure the rounding error."""
    torch.manual_seed(0)
    base = nn.Linear(64, 64, bias=False).to(torch.bfloat16)
    lin = H.LoRALinear(base, r=8, alpha=8, dropout=0.0)
    nn.init.normal_(lin.lora_B, std=0.02)
    x = torch.randn(16, 64).to(torch.bfloat16)
    ref = lin(x).float()
    merged = lin.merged_base()(x).float()
    rel = ((merged - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel


def test_lora_state_dict_roundtrip_and_strictness():
    a, b = _tiny_core(), _tiny_core()
    H.inject_lora(a, GROUPS, 0.0)
    H.inject_lora(b, GROUPS, 0.0)
    for n, p in a.named_parameters():
        if n.endswith("lora_B"):
            nn.init.normal_(p, std=0.1)
    sd = H.lora_state_dict(a)
    H.load_lora_state_dict(b, sd)
    assert all(torch.equal(sd[k], H.lora_state_dict(b)[k]) for k in sd)
    bad = dict(sd)
    bad.pop(next(iter(bad)))
    with pytest.raises(AssertionError):
        H.load_lora_state_dict(b, bad)


def test_custom_voice_config_transform_does_not_mutate_input():
    cd = {"tts_model_type": "base", "talker_config": {"x": 1}}
    out = H.to_custom_voice_config(cd, "Hindi_Female")
    assert cd["tts_model_type"] == "base" and "spk_id" not in cd["talker_config"]
    assert out["tts_model_type"] == "custom_voice"
    assert out["talker_config"]["spk_id"] == {"hindi_female": 3000}
    assert out["talker_config"]["spk_is_dialect"] == {"hindi_female": False}


def test_custom_voice_state_dict_bakes_speaker_and_drops_encoder():
    core = nn.Module()
    core.speaker_encoder = nn.Linear(2, 2)
    core.talker = nn.Module()
    core.talker.model = nn.Module()
    core.talker.model.codec_embedding = nn.Embedding(3100, 4).to(torch.bfloat16)
    spk = torch.tensor([1.0, 2.0, 3.0, 4.0])
    orig = core.talker.model.codec_embedding.weight.detach().clone()
    sd = H.custom_voice_state_dict(core, spk)
    assert not any(k.startswith("speaker_encoder") for k in sd)
    w = sd["talker.model.codec_embedding.weight"]
    assert torch.equal(w[3000], spk.to(torch.bfloat16))
    assert torch.equal(w[:3000], orig[:3000]) and torch.equal(w[3001:], orig[3001:])


# ----------------------------------------------------------------------------- export regression (the bug that broke the original run)
def _mock_base(tmp_path):
    b = tmp_path / "base"
    (b / "speech_tokenizer").mkdir(parents=True)
    for rel in ("model.safetensors", "model-00001-of-00002.safetensors", "model.safetensors.index.json", "config.json",
                "generation_config.json", "vocab.json", "merges.txt", "tokenizer_config.json", "preprocessor_config.json",
                ".gitattributes", "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors",
                "speech_tokenizer/preprocessor_config.json"):
        (b / rel).write_text("{}" if rel.endswith(".json") else "x")
    (b / "config.json").write_text(json.dumps({"tts_model_type": "base", "talker_config": {}}))
    return b


def test_original_copytree_pattern_dropped_codec_weights(tmp_path):
    """Documents the original defect: ignore_patterns matches at every depth."""
    base = _mock_base(tmp_path)
    out = tmp_path / "out_old"
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("model.safetensors", "model*.safetensors",
                                                             "model.safetensors.index.json", ".git*"))
    assert not (out / "speech_tokenizer" / "model.safetensors").exists()  # the bug


def test_copy_base_assets_keeps_codec_weights_and_skips_top_level_weights(tmp_path):
    base = _mock_base(tmp_path)
    out = tmp_path / "out"
    H.copy_base_assets(str(base), str(out))
    assert (out / "speech_tokenizer" / "model.safetensors").exists()
    for rel in ("model.safetensors", "model-00001-of-00002.safetensors", "model.safetensors.index.json", ".gitattributes"):
        assert not (out / rel).exists(), rel
    for rel in ("config.json", "generation_config.json", "vocab.json", "merges.txt", "tokenizer_config.json",
                "preprocessor_config.json", "speech_tokenizer/config.json", "speech_tokenizer/preprocessor_config.json"):
        assert (out / rel).exists(), rel


def test_verify_export_dir_detects_each_failure_mode(tmp_path):
    from safetensors.torch import save_file

    base = _mock_base(tmp_path)
    out = tmp_path / "out"
    H.copy_base_assets(str(base), str(out))
    (out / "config.json").write_text(json.dumps(H.to_custom_voice_config({"talker_config": {}}, "hindi_female")))
    probs = H.verify_export_dir(str(base), str(out), "hindi_female")
    assert probs == ["missing top-level model.safetensors"]
    save_file({"talker.model.codec_embedding.weight": torch.zeros(4, 4)}, str(out / "model.safetensors"))
    assert H.verify_export_dir(str(base), str(out), "hindi_female") == []
    # failure modes
    os.remove(out / "speech_tokenizer" / "model.safetensors")
    assert any("speech_tokenizer/model.safetensors" in p for p in H.verify_export_dir(str(base), str(out), "hindi_female"))
    save_file({"speaker_encoder.w": torch.zeros(1), "talker.model.codec_embedding.weight": torch.zeros(1)},
              str(out / "model.safetensors"))
    assert any("speaker_encoder" in p for p in H.verify_export_dir(str(base), str(out), "hindi_female"))
    save_file({"talker.x.base.weight": torch.zeros(1), "talker.model.codec_embedding.weight": torch.zeros(1)},
              str(out / "model.safetensors"))
    assert any("LoRA" in p for p in H.verify_export_dir(str(base), str(out), "hindi_female"))
    (out / "config.json").write_text(json.dumps({"tts_model_type": "base"}))
    assert any("custom_voice" in p for p in H.verify_export_dir(str(base), str(out), "hindi_female"))


# ----------------------------------------------------------------------------- real artifact (optional)
@pytest.mark.skipif(not os.environ.get("HINDI_ADAPTER_PT"), reason="set HINDI_ADAPTER_PT=/path/adapter_best.pt")
def test_real_adapter_schema():
    ck = torch.load(os.environ["HINDI_ADAPTER_PT"], map_location="cpu", weights_only=True)
    assert ck["step"] == 500 and set(ck) >= {"lora", "step", "val", "best_val", "groups"}
    sd = ck["lora"]
    assert len(sd) == 466 and sum(v.numel() for v in sd.values()) == 36_028_416
    assert all(torch.isfinite(v).all() for v in sd.values())
    pat = {g: re.compile(rx) for g, rx in H.LORA_PATTERNS.items()}
    counts = {g: 0 for g in pat}
    for k in sd:
        mod = k.rsplit(".", 1)[0]
        hit = [g for g, rx in pat.items() if rx.match(mod)]
        assert len(hit) == 1, k
        if k.endswith("lora_A"):
            counts[hit[0]] += 1
    assert counts == {"talker": 196, "text_projection": 2, "code_predictor": 35}
    assert {g: tuple(v) for g, v in ck["groups"].items()} == {"talker": (32, 32), "text_projection": (32, 32), "code_predictor": (8, 8)}
