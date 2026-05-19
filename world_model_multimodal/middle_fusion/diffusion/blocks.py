"""
blocks.py — Building blocks for the Middle-Fusion multimodal world model.

Extends the original single-modality UNet building blocks with:
  - CrossModalAttention: bidirectional cross-attention between RGB and tactile
    feature maps at the same spatial resolution (used at the bottleneck).
  - DualStreamUNet: two separate encoder streams (RGB and tactile) that run in
    parallel through the downsampling path, fuse via CrossModalAttention at the
    bottleneck, then share a single decoder that reconstructs the next RGB frame
    using RGB encoder skip connections.
"""

from functools import partial
import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Global hyperparameters
# ---------------------------------------------------------------------------

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

# ---------------------------------------------------------------------------
# Convenience conv aliases
# ---------------------------------------------------------------------------

Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class AdaGroupNorm(nn.Module):
    """FiLM conditioning: scale + shift from a conditioning vector."""

    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class SelfAttention2d(nn.Module):
    """Standard multi-head self-attention on spatial feature maps."""

    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = [t for t in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-attention between RGB and tactile feature maps at the
    same spatial resolution.

    Both directions are computed:
      * RGB attends to tactile  (RGB queries, tactile keys/values)
      * Tactile attends to RGB  (tactile queries, RGB keys/values)

    Output projections are zero-initialised so the module starts as an
    identity residual — training is stable from the first step.
    """

    def __init__(self, channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, channels // head_dim)
        assert channels % self.n_head == 0
        self._head_ch = channels // self.n_head

        # Normalisation
        self.norm_rgb = GroupNorm(channels)
        self.norm_tac = GroupNorm(channels)

        # RGB → tactile cross-attention
        self.q_rgb = Conv1x1(channels, channels)
        self.kv_tac = Conv1x1(channels, channels * 2)
        self.out_proj_rgb = Conv1x1(channels, channels)
        nn.init.zeros_(self.out_proj_rgb.weight)
        nn.init.zeros_(self.out_proj_rgb.bias)

        # Tactile → RGB cross-attention (symmetric)
        self.q_tac = Conv1x1(channels, channels)
        self.kv_rgb = Conv1x1(channels, channels * 2)
        self.out_proj_tac = Conv1x1(channels, channels)
        nn.init.zeros_(self.out_proj_tac.weight)
        nn.init.zeros_(self.out_proj_tac.bias)

    def forward(self, rgb: Tensor, tac: Tensor) -> Tuple[Tensor, Tensor]:
        n, c, h, w = rgb.shape
        nh = self.n_head
        hc = self._head_ch
        seq = h * w

        rgb_n = self.norm_rgb(rgb)
        tac_n = self.norm_tac(tac)

        # ---- RGB attends to tactile ----------------------------------------
        # queries from RGB
        q = self.q_rgb(rgb_n).view(n, nh, hc, seq).transpose(2, 3)   # (n, nh, seq, hc)
        # keys and values from tactile
        kv = self.kv_tac(tac_n).view(n, nh * 2, hc, seq).transpose(2, 3)
        k, v = kv.chunk(2, dim=1)                                       # each (n, nh, seq, hc)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hc)
        att = F.softmax(att, dim=-1)
        rgb_out = (att @ v).transpose(2, 3).reshape(n, c, h, w)
        rgb = rgb + self.out_proj_rgb(rgb_out)

        # ---- Tactile attends to RGB -----------------------------------------
        # queries from tactile
        q2 = self.q_tac(tac_n).view(n, nh, hc, seq).transpose(2, 3)
        # keys and values from RGB
        kv2 = self.kv_rgb(rgb_n).view(n, nh * 2, hc, seq).transpose(2, 3)
        k2, v2 = kv2.chunk(2, dim=1)
        att2 = (q2 @ k2.transpose(-2, -1)) / math.sqrt(hc)
        att2 = F.softmax(att2, dim=-1)
        tac_out = (att2 @ v2).transpose(2, 3).reshape(n, c, h, w)
        tac = tac + self.out_proj_tac(tac_out)

        return rgb, tac


# ---------------------------------------------------------------------------
# Noise-level embedding
# ---------------------------------------------------------------------------


class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: Tensor) -> Tensor:
        assert input.ndim == 1
        f = 2 * math.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


# ---------------------------------------------------------------------------
# Sampling layers
# ---------------------------------------------------------------------------


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """
    Residual block with FiLM conditioning via AdaGroupNorm and optional
    self-attention.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, attn: bool) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r
        x = self.attn(x)
        return x


class ResBlocks(nn.Module):
    """Sequence of ResBlocks with optional skip-connection concatenation."""

    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_ch, out_ch, cond_channels, attn)
                for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
            ]
        )

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        to_cat: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x, cond)
            outputs.append(x)
        return x, outputs


# ---------------------------------------------------------------------------
# Single-stream UNet (unchanged from original)
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """
    Standard UNet with encoder, mid-block, and decoder.
    Used as the building block for each stream in DualStreamUNet.
    """

    def __init__(
        self,
        cond_channels: int,
        depths: List[int],
        channels: List[int],
        attn_depths: List[int],
    ) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self._num_down = len(channels) - 1

        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=bool(attn_depths[i]),
                )
            )
            u_blocks.append(
                ResBlocks(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    attn=bool(attn_depths[i]),
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(list(reversed(u_blocks)))

        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=True,
        )

        downsamples = [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        upsamples = [nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])]
        self.downsamples = nn.ModuleList(downsamples)
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, x: Tensor, cond: Tensor) -> Tuple[Tensor, List, List]:
        *_, h, w = x.size()
        n = self._num_down
        padding_h = math.ceil(h / 2 ** n) * 2 ** n - h
        padding_w = math.ceil(w / 2 ** n) * 2 ** n - w
        x = F.pad(x, (0, padding_w, 0, padding_h))

        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append((x_down, *block_outputs))

        x, _ = self.mid_blocks(x, cond)

        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, block_outputs = block(x_up, cond, skip[::-1])
            u_outputs.append((x_up, *block_outputs))

        x = x[..., :h, :w]
        return x, d_outputs, u_outputs


# ---------------------------------------------------------------------------
# Dual-stream UNet with middle fusion
# ---------------------------------------------------------------------------


class DualStreamUNet(nn.Module):
    """
    Dual-stream UNet for middle fusion of two image modalities.

    Architecture:
      - RGB encoder: separate d_blocks + downsamples
      - Tactile encoder: separate d_blocks + downsamples (same topology)
      - Shared mid_blocks (applied to RGB; tactile mid_blocks are separate but
        same architecture so each modality has its own mid representation)
      - CrossModalAttention: fuses RGB and tactile at the bottleneck
      - Shared decoder (u_blocks + upsamples): uses fused RGB features and RGB
        encoder skip connections to reconstruct the next RGB frame.

    The decoder only uses RGB encoder skip connections because the prediction
    target is the next RGB frame.
    """

    def __init__(
        self,
        cond_channels: int,
        depths: List[int],
        channels: List[int],
        attn_depths: List[int],
    ) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self._num_down = len(channels) - 1

        # ---- RGB encoder ---------------------------------------------------
        rgb_d_blocks, rgb_u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            rgb_d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=bool(attn_depths[i]),
                )
            )
            # Decoder u_blocks: the first ResBlock in each decoder stage
            # receives the upsampled feature concatenated with the RGB encoder
            # skip at that level.
            rgb_u_blocks.append(
                ResBlocks(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    attn=bool(attn_depths[i]),
                )
            )
        self.rgb_d_blocks = nn.ModuleList(rgb_d_blocks)
        self.rgb_u_blocks = nn.ModuleList(list(reversed(rgb_u_blocks)))

        rgb_downsamples = [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        rgb_upsamples = [nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])]
        self.rgb_downsamples = nn.ModuleList(rgb_downsamples)
        self.rgb_upsamples = nn.ModuleList(rgb_upsamples)

        # ---- Tactile encoder -----------------------------------------------
        tac_d_blocks = []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            tac_d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=bool(attn_depths[i]),
                )
            )
        self.tac_d_blocks = nn.ModuleList(tac_d_blocks)

        tac_downsamples = [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        self.tac_downsamples = nn.ModuleList(tac_downsamples)

        # ---- Mid-blocks (one per stream, fused after) ----------------------
        self.rgb_mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=True,
        )
        self.tac_mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=True,
        )

        # ---- Cross-modal attention at bottleneck ---------------------------
        self.cross_modal_attn = CrossModalAttention(channels[-1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad(self, x: Tensor) -> Tensor:
        *_, h, w = x.size()
        n = self._num_down
        pad_h = math.ceil(h / 2 ** n) * 2 ** n - h
        pad_w = math.ceil(w / 2 ** n) * 2 ** n - w
        return F.pad(x, (0, pad_w, 0, pad_h))

    def encode_rgb(
        self, x: Tensor, cond: Tensor
    ) -> Tuple[Tensor, List]:
        """Run the RGB encoder stream; return bottleneck features and skip list."""
        x = self._pad(x)
        d_outputs: List = []
        for block, down in zip(self.rgb_d_blocks, self.rgb_downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append((x_down, *block_outputs))
        x, _ = self.rgb_mid_blocks(x, cond)
        return x, d_outputs

    def encode_tac(self, x: Tensor, cond: Tensor) -> Tensor:
        """Run the tactile encoder stream; return bottleneck features only."""
        x = self._pad(x)
        for block, down in zip(self.tac_d_blocks, self.tac_downsamples):
            x_down = down(x)
            x, _ = block(x_down, cond)
        x, _ = self.tac_mid_blocks(x, cond)
        return x

    def decode(
        self,
        x: Tensor,
        cond: Tensor,
        rgb_d_outputs: List,
        orig_h: int,
        orig_w: int,
    ) -> Tensor:
        """Run the shared decoder using fused bottleneck + RGB skip connections."""
        for block, up, skip in zip(
            self.rgb_u_blocks, self.rgb_upsamples, reversed(rgb_d_outputs)
        ):
            x_up = up(x)
            x, _ = block(x_up, cond, skip[::-1])
        x = x[..., :orig_h, :orig_w]
        return x

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(
        self, rgb: Tensor, tac: Tensor, cond: Tensor
    ) -> Tensor:
        """
        Args:
            rgb:  (B, rgb_in_ch, H, W)  — concatenated past RGB frames + noisy next
            tac:  (B, tac_in_ch, H, W)  — concatenated past tactile frames
            cond: (B, cond_channels)    — global conditioning vector

        Returns:
            (B, channels[0], H, W)  — decoded features ready for conv_out
        """
        *_, orig_h, orig_w = rgb.shape

        # Encode both streams
        rgb_bottleneck, rgb_d_outputs = self.encode_rgb(rgb, cond)
        tac_bottleneck = self.encode_tac(tac, cond)

        # Fuse at bottleneck via cross-modal attention
        rgb_fused, _ = self.cross_modal_attn(rgb_bottleneck, tac_bottleneck)

        # Decode using fused RGB features and RGB skip connections
        out = self.decode(rgb_fused, cond, rgb_d_outputs, orig_h, orig_w)
        return out
