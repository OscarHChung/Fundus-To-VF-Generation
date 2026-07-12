"""Swappable frozen-backbone loader (Task A1 of the sub-4.0 encoder plan).

One responsibility: given an `--encoder` name, return a frozen module that exposes

    enc.encode_prefix(imgs: Tensor[B,3,H,W], input_size: int = 224) -> Tensor[B, 1+P, D]
        # CLS token at index 0, then row-major patch tokens. input_size selects the patch grid to
        # run the (RETFound-MAE) forward at; default 224 reproduces the original fixed-224 forward
        # byte-for-byte. For RETFound-MAE, any input_size != 224 must be a multiple of 16 (the
        # patch16 grid) or a clear ValueError is raised. Other backbones accept-and-ignore this
        # kwarg, always running at their configured enc.input_size.
    enc.grid -> (gh, gw)   (P = gh*gw)      enc.dim -> int      enc.input_size -> int

so the rest of the pipeline can treat every backbone uniformly. This isolates all backbone-specific
loading/preprocessing from training.py.

Backbones
- retfound_mae  : the CURRENT model's frozen ViT-L/16 @224. Reproduces training._encode EXACTLY
  (frozen forward, NO MAE random patch shuffle, enc.norm applied). Reuses the already-loaded
  training.base_model global so we never hold the 3.7 GB checkpoint twice on this 16 GB box.
- dinov2_l / dinov3_l / retfound_dinov2 : ViT-L/14 @224 (grid 16x16, D=1024), register tokens dropped.
  Loaded on demand via HuggingFace transformers; absent weights raise at load_encoder() time only.
- visionfm : retinal foundation model (loaded from its published weights if present).
- retizero : RetiZero (Wang et al. 2025, Nat Commun) — vision-language ViT-L/16, a LoRA rank-8
  (q,v only) fine-tune of a RETFound-style MAE ViT-L/16 against a CLIP text objective over 341k
  fundus image-text pairs. Public Google Drive weights (github.com/LooKing9218/RetiZero), no
  login/access-request gate. Loaded from local weights if present (see _load_retizero docstring).

Memory rule: exactly one torch process at a time; every backbone is frozen + .eval().
"""
import os
import sys
import torch
import torch.nn as nn

# HF repo ids for the on-demand backbones (all public).
_HF_IDS = {
    "dinov2_l":        "facebook/dinov2-large",
    "dinov3_l":        "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "retfound_dinov2": "YukunZhou/RETFound_dinov2_meh",
}

# encoder/RETFound_MAE/util/pos_embed.py — needed to regenerate the sin-cos pos-embed at a
# non-224 grid (see _mae_prefix below). Added to sys.path lazily (not at import time) so this
# module stays cheap to import when only the DINOv2/VisionFM backbones are needed.
_RETFOUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "encoder", "RETFound_MAE")


class FrozenEncoder(nn.Module):
    """Frozen backbone with a uniform prefix interface. Never trained."""

    def __init__(self, backbone, grid, dim, input_size, prefix_fn):
        super().__init__()
        self.backbone = backbone.eval()
        self.grid = tuple(grid)
        self.dim = int(dim)
        self.input_size = int(input_size)
        self._prefix_fn = prefix_fn
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_prefix(self, imgs, input_size=224):
        """(B,3,H,W) -> (B, 1+gh*gw, D). CLS/first token at index 0, then row-major patch tokens.

        input_size selects the patch grid to run the (RETFound-MAE) forward at; the default 224
        reproduces the original fixed-224 forward byte-for-byte (see _mae_prefix). For RETFound-MAE,
        any input_size != 224 must be a multiple of 16 (the patch16 grid) — otherwise _mae_prefix
        raises a clear ValueError instead of failing deep inside a shape-mismatched tensor op. Other
        backbones currently ignore this kwarg and always run at their configured `self.input_size`."""
        return self._prefix_fn(self.backbone, imgs, input_size)


# ----------------------------------------------------------------------------- RETFound-MAE
def _mae_prefix(backbone, imgs, input_size=224):
    """Frozen ViT-L/16 forward, byte-identical to training.PerPointVFModel._encode (frozen branch)
    AT input_size==224 (the default): patch-embed, add pos-embed, prepend (cls + pos), all 24
    blocks, then enc.norm. No random_masking. This 224 branch is UNCHANGED from before high-res
    support was added — test_default_224_byte_identical (decoder/tests_bakeoff_highres.py) guards it.

    For input_size != 224, timm's PatchEmbed.forward hard-asserts H==img_size, so we bypass it:
    run the inner Conv2d projection directly (kernel16/stride16 — works at any size divisible by
    16) and regenerate the sin-cos positional embedding at the new grid via get_2d_sincos_pos_embed.
    Every other op mirrors the 224 forward exactly (same cls-token handling, same blocks, same norm)."""
    if input_size == 224:
        h = backbone.patch_embed(imgs)
        h = h + backbone.pos_embed[:, 1:, :]
        cls = (backbone.cls_token + backbone.pos_embed[:, :1, :]).expand(h.shape[0], -1, -1)
        h = torch.cat((cls, h), dim=1)
        for blk in backbone.blocks:
            h = blk(h)
        return backbone.norm(h)
    if input_size % 16 != 0:
        raise ValueError(
            f"input_size must be a multiple of 16 (RETFound-MAE patch16 grid); got {input_size}"
        )
    if _RETFOUND_DIR not in sys.path:
        sys.path.insert(0, _RETFOUND_DIR)
    from util.pos_embed import get_2d_sincos_pos_embed
    h = backbone.patch_embed.proj(imgs).flatten(2).transpose(1, 2)      # (B, P, D); Conv2d k16/s16
    grid_size = input_size // 16
    embed_dim = backbone.pos_embed.shape[-1]
    pos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)  # (1+P, D) numpy, float32
    pos = torch.from_numpy(pos).float().unsqueeze(0).to(device=h.device, dtype=h.dtype)  # (1,1+P,D)
    h = h + pos[:, 1:, :]
    cls = (backbone.cls_token + pos[:, :1, :]).expand(h.shape[0], -1, -1)
    h = torch.cat((cls, h), dim=1)
    for blk in backbone.blocks:
        h = blk(h)
    return backbone.norm(h)


def _load_retfound_mae():
    # Reuse the module-global base_model that training.py already loaded (one 3.7 GB read, one copy).
    import training as T
    return FrozenEncoder(T.base_model, grid=(14, 14), dim=1024, input_size=224, prefix_fn=_mae_prefix)


# ----------------------------------------------------------------------------- DINOv2 family
def _dinov2_prefix(backbone, imgs, input_size=224):
    """HF Dinov2/Dinov3 model -> (B, 1+P, D). last_hidden_state is [CLS, (registers), patches];
    drop any register tokens so the output is exactly [CLS] + gh*gw row-major patch tokens.
    input_size is accepted-and-ignored (ONLY the RETFound-MAE prefix honours it): DINOv2/DINOv3
    always run at whatever resolution the loader configured (see _load_dinov2_family)."""
    n_reg = int(getattr(backbone.config, "num_register_tokens", 0) or 0)
    out = backbone(pixel_values=imgs).last_hidden_state          # (B, 1+n_reg+P, D)
    if n_reg > 0:
        out = torch.cat([out[:, :1, :], out[:, 1 + n_reg:, :]], dim=1)
    return out


def _load_dinov2_family(name, input_size=224):
    """DINOv2/v3 interpolate their position embeddings, so we run them at OUR pipeline resolution
    (224, ImageNet-norm) for an apples-to-apples comparison with RETFound-MAE — NOT the model's
    native cfg.image_size (dinov2-large is 518). Grid is therefore derived from 224, not the config."""
    from transformers import AutoModel
    hf_id = _HF_IDS[name]
    backbone = AutoModel.from_pretrained(hf_id)
    cfg = backbone.config
    dim = int(cfg.hidden_size)
    patch = int(getattr(cfg, "patch_size", 14))
    g = input_size // patch
    return FrozenEncoder(backbone, grid=(g, g), dim=dim, input_size=input_size, prefix_fn=_dinov2_prefix)


# ----------------------------------------------------------------------------- VisionFM
def _load_visionfm():
    """VisionFM retinal ViT-B/16 @224. Loaded from local weights if present (no public HF loader);
    raises a clear error otherwise so the bake-off records it as unavailable rather than crashing."""
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "encoder", "VisionFM_retinal.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"VisionFM weights not found at {ckpt}; download to enable this encoder")
    import timm
    backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model", state.get("teacher", state))
    backbone.load_state_dict({k.replace("backbone.", ""): v for k, v in state.items()}, strict=False)

    def _prefix(bb, imgs, input_size=224):
        h = bb.patch_embed(imgs)
        cls = bb.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat((cls, h), dim=1) + bb.pos_embed
        h = bb.pos_drop(h)
        for blk in bb.blocks:
            h = blk(h)
        return bb.norm(h)

    return FrozenEncoder(backbone, grid=(14, 14), dim=768, input_size=224, prefix_fn=_prefix)


# ----------------------------------------------------------------------------- RetiZero (LoRA-CLIP ViT-L/16)
_RETIZERO_CKPT_ENV = "RETIZERO_CKPT"
_RETIZERO_DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "encoder", "RetiZero.pth")
_RETIZERO_GDRIVE_URL = "https://drive.google.com/file/d/14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM/view"


class _LoRA_qkv(nn.Module):
    """Wraps a timm ViT block's `attn.qkv` Linear with RetiZero's rank-8 LoRA deltas on Q and V only
    (K is untouched) — reproduces LooKing9218/RetiZero's clip_modules/modeling/LORA/lora_image_encoder.py
    `_LoRA_qkv` exactly. State-dict names (qkv.{weight,bias}, linear_a_q, linear_b_q, linear_a_v,
    linear_b_v; the four LoRA linears have no bias) were verified against the released checkpoint."""

    def __init__(self, qkv, r=8):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.linear_a_q = nn.Linear(self.dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, self.dim, bias=False)

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv = qkv.clone()
        qkv[:, :, :self.dim] = qkv[:, :, :self.dim] + new_q
        qkv[:, :, -self.dim:] = qkv[:, :, -self.dim:] + new_v
        return qkv


def _retizero_prefix(backbone, imgs, input_size=224):
    """Frozen LoRA-CLIP ViT-L/16 forward -> (B, 1+196, D). Mirrors _mae_prefix's 224 branch and
    _load_visionfm's _prefix: timm PatchEmbed, prepend cls_token, add pos_embed, run all 24
    (LoRA-patched) blocks, final norm over the WHOLE sequence. Deliberately bypasses RetiZero's own
    CLS-only classification pooling and its CLIP projection_head_vision (both are output-side heads
    for their own tasks) — we want raw token-level features, uniform with every other backbone here.
    input_size is accepted-and-ignored: RetiZero's checkpoint only supports its native 224 grid
    (a LoRA fine-tune of a fixed-resolution backbone, unlike RETFound-MAE's sin-cos regeneration)."""
    h = backbone.patch_embed(imgs)
    cls = backbone.cls_token.expand(h.shape[0], -1, -1)
    h = torch.cat((cls, h), dim=1) + backbone.pos_embed
    h = backbone.pos_drop(h)
    for blk in backbone.blocks:
        h = blk(h)
    return backbone.norm(h)


def _load_retizero():
    """RetiZero ungated weights: public Google Drive link in the repo README
    (github.com/LooKing9218/RetiZero), no login/access-request gate — verified by downloading the
    full 1.65 GB checkpoint via `gdown` with no authentication (only Google's automatic large-file
    'cannot scan for viruses' confirm-token redirect, not an auth wall). No LICENSE file in the
    repo (all-rights-reserved by default; the paper describes research use) — noted here, not
    enforced. Weights expected at encoder/RetiZero.pth (not checked into git; encoder/*.pth is
    gitignored) or at the path in $RETIZERO_CKPT.

    Checkpoint layout (verified): a CLIP-style state_dict with 'vision_model.model.lora_vit.*'
    (RETFound ViT-L/16, LoRA rank-8 on q/v, depth 24, D=1024, patch16@224 — architecturally
    identical to timm's vit_large_patch16_224), 'vision_model.projection_head_vision.*' (CLIP joint
    -space head, unused here), 'text_model.*' (BERT text tower, unused here), and 'logit_scale'."""
    ckpt_path = os.environ.get(_RETIZERO_CKPT_ENV, _RETIZERO_DEFAULT_CKPT)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"RetiZero weights not found at {ckpt_path}; download the public (ungated) checkpoint "
            f"from {_RETIZERO_GDRIVE_URL} (linked from https://github.com/LooKing9218/RetiZero) and "
            f"place it there, or set ${_RETIZERO_CKPT_ENV}."
        )
    import timm
    backbone = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    for blk in backbone.blocks:
        blk.attn.qkv = _LoRA_qkv(blk.attn.qkv, r=8)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prefix = "vision_model.model.lora_vit."
    vit_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = backbone.load_state_dict(vit_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"RetiZero state_dict mismatch (checkpoint layout changed?): missing={missing} "
            f"unexpected={unexpected}"
        )
    return FrozenEncoder(backbone, grid=(14, 14), dim=1024, input_size=224, prefix_fn=_retizero_prefix)


# ----------------------------------------------------------------------------- RETFound-Green
_GREEN_CKPT_ENV = "RETFOUND_GREEN_CKPT"
_GREEN_DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "encoder", "RETFound_Green.pth")
_GREEN_RELEASE_URL = ("https://github.com/justinengelmann/RETFound_Green/releases/download/"
                       "v0.1/retfoundgreen_statedict.pth")


def _green_prefix(backbone, imgs, input_size=392):
    """RETFound-Green (ViT-S/14 + 4 register tokens) forward -> (B, 1+P, D). Registers are dropped
    so the output is exactly [CLS] + row-major patch tokens, uniform with every other backbone here.
    input_size is accepted-and-ignored: the checkpoint is fixed at 392x392 (img_size baked in at
    timm.create_model() time in _load_retfound_green); callers must feed 392x392 images themselves
    (this backbone does NOT regenerate its pos-embed at other resolutions, unlike RETFound-MAE)."""
    out = backbone.forward_features(imgs)                  # (B, 1+4+P, D): [CLS, reg x4, patches]
    n_reg = int(getattr(backbone, "num_prefix_tokens", 1)) - 1
    if n_reg > 0:
        out = torch.cat([out[:, :1, :], out[:, 1 + n_reg:, :]], dim=1)
    return out


def _load_retfound_green():
    """RETFound-Green (Engelmann et al. 2025, Nat Commun; justinengelmann/RETFound_Green) —
    ViT-Small/14 + 4 register tokens, DINOv2-style Token-Reconstruction pretraining on 75k public
    images, native 392x392, D=384. Ungated: a plain public GitHub Release binary (`wget`, no login/
    access request) — retfoundgreen_statedict.pth at the URL below. No LICENSE file found in the
    repo at time of writing (noted, not enforced).

    Preprocessing NOTE (caller's responsibility, NOT enforced here): the released model expects
    392x392 images normalized with mean=std=0.5 per channel (NOT ImageNet stats) — different from
    every other backbone in this module. A generic ImageNet-normalizing cache loop (e.g. the
    diag_encoder_bakeoff non-default-config branch) would silently miscalibrate this encoder's
    inputs; use a dedicated transform for this encoder.

    Requires timm>=0.9.12 (this project pins 0.6.13 for the other backbones here, which predates
    the *_reg4_dinov2 architecture family; upgrading is safe — encoders.py is the only module in
    this codebase that imports timm, and the vit_base_patch16_224/vit_large_patch16_224 architectures
    used elsewhere in this file are unchanged across that version range, verified). Raises a clear,
    actionable ImportError if the installed timm is too old rather than a deep error inside
    timm.create_model. Weights expected at encoder/RETFound_Green.pth (not checked into git) or at
    $RETFOUND_GREEN_CKPT."""
    ckpt_path = os.environ.get(_GREEN_CKPT_ENV, _GREEN_DEFAULT_CKPT)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"RETFound-Green weights not found at {ckpt_path}; download the public (ungated) "
            f"checkpoint with `wget {_GREEN_RELEASE_URL}` and place it there, or set "
            f"${_GREEN_CKPT_ENV}."
        )
    import timm
    ver = tuple(int(p) for p in timm.__version__.split(".")[:2])
    if ver < (0, 9):
        raise ImportError(
            f"RETFound-Green needs timm>=0.9.12 (vit_small_patch14_reg4_dinov2 architecture); "
            f"found timm=={timm.__version__}. `pip install -U timm` (safe here; see docstring)."
        )
    backbone = timm.create_model(
        "vit_small_patch14_reg4_dinov2", img_size=(392, 392), num_classes=0,
        checkpoint_path=ckpt_path,
    )
    return FrozenEncoder(backbone, grid=(28, 28), dim=384, input_size=392, prefix_fn=_green_prefix)


# ----------------------------------------------------------------------------- dispatch
_LOADERS = {
    "retfound_mae":    _load_retfound_mae,
    "dinov2_l":        lambda: _load_dinov2_family("dinov2_l"),
    "dinov3_l":        lambda: _load_dinov2_family("dinov3_l"),
    "retfound_dinov2": lambda: _load_dinov2_family("retfound_dinov2"),
    "visionfm":        _load_visionfm,
    "retizero":        _load_retizero,
    "retfound_green":  _load_retfound_green,
}


def load_encoder(name):
    if name not in _LOADERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(_LOADERS)}")
    return _LOADERS[name]()
