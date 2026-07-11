"""Swappable frozen-backbone loader (Task A1 of the sub-4.0 encoder plan).

One responsibility: given an `--encoder` name, return a frozen module that exposes

    enc.encode_prefix(imgs: Tensor[B,3,H,W]) -> Tensor[B, 1+P, D]   # CLS token at index 0
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
        reproduces the original fixed-224 forward byte-for-byte (see _mae_prefix). Other backbones
        currently ignore this kwarg and always run at their configured `self.input_size`."""
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


# ----------------------------------------------------------------------------- dispatch
_LOADERS = {
    "retfound_mae":    _load_retfound_mae,
    "dinov2_l":        lambda: _load_dinov2_family("dinov2_l"),
    "dinov3_l":        lambda: _load_dinov2_family("dinov3_l"),
    "retfound_dinov2": lambda: _load_dinov2_family("retfound_dinov2"),
    "visionfm":        _load_visionfm,
}


def load_encoder(name):
    if name not in _LOADERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(_LOADERS)}")
    return _LOADERS[name]()
