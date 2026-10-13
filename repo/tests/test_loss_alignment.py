"""Equivalence/correctness tests for the custom single-shift loss (training/htts_common.py::compute_loss).

These run on a tiny random model with the real qwen-tts module structure (CPU, seconds). They demonstrate:
  1. main (codebook-0) loss == the library's own loss when the library is given UNSHIFTED labels
     (i.e. exactly one shift, as HF intends);
  2. the legacy pattern used by the official/community scripts (manual shift + library loss) equals a
     TWO-step-ahead cross-entropy -> the "double shift" defect is real and reproducible;
  3. the library's forward_sub_talker_finetune double-shifts and cannot express the correct alignment;
     our sub-talker CE equals the library loss on the codebooks it can express and covers one more;
  4. hidden-state pairing: frame p uses the hidden state at position p-1 (not p).
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qwen_tts")
import htts_common as H
import torch.nn.functional as F
from tiny_model import NUM_GROUPS, build_tiny_core, make_batch


@pytest.fixture(scope="module")
def setup():
    core, hf_cfg = build_tiny_core()
    batch, rows = make_batch(hf_cfg)
    spk = torch.randn(32)
    return core, hf_cfg, batch, rows, spk


def test_main_loss_equals_library_single_shift(setup):
    core, _, batch, _, spk = setup
    talker = core.talker
    with torch.no_grad():
        total, st = H.compute_loss(core, batch, spk, sub_weight=0.3)
        x = H.build_talker_inputs(core, batch, spk)
        # library path, UNSHIFTED labels, full-length sequence: ForCausalLMLoss shifts exactly once
        lib = talker(inputs_embeds=x, attention_mask=batch["attn"], labels=batch["labels"]).loss
    assert torch.allclose(st["main"].float(), lib.float(), atol=1e-5), (st["main"].item(), lib.item())


def test_legacy_pattern_is_two_steps_ahead(setup):
    core, _, batch, _, spk = setup
    talker = core.talker
    with torch.no_grad():
        x = H.build_talker_inputs(core, batch, spk)
        legacy = talker(inputs_embeds=x[:, :-1], attention_mask=batch["attn"][:, :-1],
                        labels=batch["labels"][:, 1:]).loss  # manual shift + library shift
        logits = talker(inputs_embeds=x[:, :-1], attention_mask=batch["attn"][:, :-1]).logits.float()
        lab = batch["labels"]
        # position p predicts label[p+2] under the double shift
        tgt2 = torch.full_like(lab[:, 2:], -100)
        tgt2[:] = lab[:, 2:]
        manual = F.cross_entropy(logits[:, : tgt2.shape[1]].reshape(-1, logits.shape[-1]), tgt2.reshape(-1),
                                 ignore_index=-100)
        _, st = H.compute_loss(core, batch, spk, 0.3)
    assert torch.allclose(legacy.float(), manual, atol=1e-5)
    assert abs(legacy.item() - st["main"].item()) > 1e-4  # and it differs from the corrected loss


def test_sub_talker_alignment(setup):
    core, _, batch, _, spk = setup
    talker = core.talker
    with torch.no_grad():
        x = H.build_talker_inputs(core, batch, spk)
        out = talker(inputs_embeds=x[:, :-1], attention_mask=batch["attn"][:, :-1], output_hidden_states=True)
        hidden = out.hidden_states[0][-1]
        fh = hidden[batch["codec_mask"][:, 1:]]
        fc = batch["codec_ids"][batch["codec_mask"]]
        sub_in = H._sub_talker_inputs(talker, fh, fc)
        logits = talker.code_predictor.forward_finetune(inputs_embeds=sub_in).logits.float()  # (F, G-1, V)
        G1 = NUM_GROUPS - 1
        assert logits.shape[1] == G1
        # ours: column j predicts codebook j+1, all G-1 columns
        ours_all = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), fc[:, 1:].reshape(-1))
        # library with labels=fc[:, :G-1]: after ITS shift, column j is graded against fc[:, j+1] for j < G-2
        lib = talker.code_predictor.forward_finetune(inputs_embeds=sub_in, labels=fc[:, :G1]).loss
        ours_first = F.cross_entropy(logits[:, : G1 - 1].reshape(-1, logits.shape[-1]), fc[:, 1 : G1].reshape(-1))
        # the library's own helper (what legacy used) grades column j against codebook j+2
        _, legacy = talker.forward_sub_talker_finetune(fc, fh)
        manual_legacy = F.cross_entropy(logits[:, : G1 - 1].reshape(-1, logits.shape[-1]), fc[:, 2:].reshape(-1))
        _, st = H.compute_loss(core, batch, spk, 0.3)
    assert torch.allclose(lib.float(), ours_first, atol=1e-5)          # same alignment on shared columns
    assert torch.allclose(legacy.float(), manual_legacy, atol=1e-5)     # legacy helper == two-ahead
    assert torch.allclose(st["sub"].float(), ours_all, atol=1e-5)       # compute_loss uses all G-1 columns
    assert abs(legacy.item() - ours_all.item()) > 1e-4


def test_hidden_state_pairing_is_p_minus_1(setup):
    _, _, batch, rows, _ = setup
    B, T = batch["codec_mask"].shape
    fake_hidden = torch.arange(T).view(1, T, 1).expand(B, T, 1).float()[:, :-1]  # hidden[b, t] == t
    picked = fake_hidden[batch["codec_mask"][:, 1:]].view(-1)
    expected = []
    for i, r in enumerate(rows):
        L, N = len(r["text_ids"]), len(r["audio_codes"])
        s = 8 + L - 1
        expected += [float(p - 1) for p in range(s, s + N)]
    assert picked.tolist() == expected


def test_diagnostic_corrected_not_worse_than_legacy_shape(setup):
    """compute_legacy_loss must run and return scalars (used by the step-0 diagnostic)."""
    core, _, batch, _, spk = setup
    lm, ls = H.compute_legacy_loss(core, batch, spk)
    assert lm.dim() == 0 and ls.dim() == 0
