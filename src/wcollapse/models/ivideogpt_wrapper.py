"""World-model wrapper.

The interface is the slim one the rest of the pipeline depends on:

    latent = wm.encode(rgb)                  # (B, T, H, W, 3) uint8 -> (B, T, D)
    next_latent = wm.predict_next(latent, a) # one-step latent dynamics
    rgb_hat = wm.decode(latent)              # for pixel-error metrics
    loss = wm.compute_loss(batch)            # reconstruction + dynamics

Two backbones:

* ``mini``       — small CNN encoder + dense latent + MLP residual dynamics +
                   CNN decoder. CPU-friendly; used by the smoke config.
* ``ivideogpt``  — loads the pretrained ``thuml/ivideogpt-oxe-64-act-free``
                   tokenizer (VQ-VAE on robotic manipulation pixels) and uses
                   its continuous pre-quantization features as the visual
                   latent. Dynamics is still our learned MLP — we don't drive
                   the autoregressive transformer in the actor's gradient
                   path because the discrete-token rollout doesn't admit a
                   reparameterized policy gradient. See plan §Module 4.

Both backbones share the same outer interface, so the Dreamer-style
imagination loop, reward/semantic heads, and evaluation code don't care
which one is active.
"""

from __future__ import annotations

import os
import sys as _sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# iVideoGPT lives in a sibling submodule; we don't pip-install it because the
# upstream repo has no setup.py. Adding the package root to sys.path lets the
# wrapper import its modules without modifying the upstream tree.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_IVIDEOGPT_PATH = os.path.join(_REPO_ROOT, "iVideoGPT")
if os.path.isdir(_IVIDEOGPT_PATH) and _IVIDEOGPT_PATH not in _sys.path:
    _sys.path.insert(0, _IVIDEOGPT_PATH)


@dataclass
class WMConfig:
    image_size: int = 64
    latent_dim: int = 128
    action_dim: int = 4
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32)
    dynamics_hidden: int = 256
    dynamics_layers: int = 2
    free_nats: float = 1.0  # placeholder for KL-style regularization if we add it
    backbone: Literal["mini", "ivideogpt"] = "mini"

    # Variant C: train the iVideoGPT tokenizer during Phase B (matches
    # iVideoGPT/mbrl/video_predictor.py:114-126 "selected_params" behaviour
    # — every tokenizer param trains EXCEPT the codebook embedding, so the
    # codebook stays a stable anchor and codebook collapse doesn't
    # confound the WM-collapse measurement).
    unfreeze_tokenizer: bool = False
    freeze_codebook_only: bool = True


def _rgb_to_tensor(rgb: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 HxWx3 or BxHxWx3 or BxTxHxWx3 -> float CHW tensor in [-1, 1]."""
    if isinstance(rgb, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(rgb))
    else:
        x = rgb
    x = x.to(device).float() / 127.5 - 1.0
    if x.dim() == 3:  # H, W, 3
        x = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # 1,1,3,H,W
    elif x.dim() == 4:  # B, H, W, 3
        x = x.permute(0, 3, 1, 2).unsqueeze(1)  # B,1,3,H,W
    elif x.dim() == 5:  # B, T, H, W, 3
        x = x.permute(0, 1, 4, 2, 3)  # B,T,3,H,W
    else:
        raise ValueError(f"Unexpected rgb shape {tuple(x.shape)}")
    return x


class _ConvEncoder(nn.Module):
    def __init__(self, image_size: int, channels: tuple[int, ...], latent_dim: int):
        super().__init__()
        layers = []
        in_c = 3
        for out_c in channels:
            layers += [nn.Conv2d(in_c, out_c, 4, 2, 1), nn.SiLU()]
            in_c = out_c
        self.conv = nn.Sequential(*layers)
        # After len(channels) strides of 2, spatial = image_size / 2**len.
        n_strides = len(channels)
        self.out_spatial = max(image_size // (2**n_strides), 1)
        flat = in_c * self.out_spatial * self.out_spatial
        self.fc = nn.Linear(flat, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, 3, H, W)
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


class _ConvDecoder(nn.Module):
    def __init__(self, image_size: int, channels: tuple[int, ...], latent_dim: int):
        super().__init__()
        n_strides = len(channels)
        self.in_spatial = max(image_size // (2**n_strides), 1)
        self.in_channels = channels[0]
        self.fc = nn.Linear(latent_dim, self.in_channels * self.in_spatial * self.in_spatial)
        layers = []
        in_c = channels[0]
        for out_c in channels[1:]:
            layers += [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.SiLU()]
            in_c = out_c
        layers += [nn.ConvTranspose2d(in_c, 3, 4, 2, 1)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.in_channels, self.in_spatial, self.in_spatial)
        return torch.tanh(self.deconv(h))


class _Dynamics(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int, layers: int):
        super().__init__()
        dims = [latent_dim + action_dim] + [hidden] * layers
        net = []
        for i in range(len(dims) - 1):
            net += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        net += [nn.Linear(dims[-1], latent_dim)]
        self.net = nn.Sequential(*net)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Residual update: z_{t+1} = z_t + f(z_t, a_t).
        delta = self.net(torch.cat([latent, action], dim=-1))
        return latent + delta


class MiniWorldModel(nn.Module):
    """Stub world model. Replace with iVideoGPT wrapper once vendored."""

    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = _ConvEncoder(cfg.image_size, cfg.encoder_channels, cfg.latent_dim)
        self.decoder = _ConvDecoder(cfg.image_size, cfg.decoder_channels, cfg.latent_dim)
        self.dynamics = _Dynamics(
            cfg.latent_dim, cfg.action_dim, cfg.dynamics_hidden, cfg.dynamics_layers
        )

    @property
    def latent_dim(self) -> int:
        return self.cfg.latent_dim

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ---------- inference helpers ----------
    @torch.no_grad()
    def encode_frame(self, rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode a single frame or batch of frames. Returns (B, D)."""
        x = _rgb_to_tensor(rgb, self.device)  # (B, T, 3, H, W)
        B, T = x.shape[:2]
        z = self.encoder(x.reshape(B * T, *x.shape[2:]))
        return z.view(B, T, -1)[:, -1]

    @torch.no_grad()
    def rollout_latent(
        self, init_latent: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Roll out the latent dynamics under a sequence of actions.

        init_latent: (B, D)
        actions:     (B, H, action_dim)
        returns:     (B, H+1, D) including the initial latent.
        """
        zs = [init_latent]
        z = init_latent
        for h in range(actions.shape[1]):
            z = self.dynamics(z, actions[:, h])
            zs.append(z)
        return torch.stack(zs, dim=1)

    @torch.no_grad()
    def decode_to_rgb(self, latent: torch.Tensor) -> np.ndarray:
        """Decode latent (B, D) or (B, T, D) to uint8 RGB (B, ..., H, W, 3)."""
        flat_in = latent.dim() == 2
        if flat_in:
            latent_in = latent
        else:
            B, T = latent.shape[:2]
            latent_in = latent.reshape(B * T, -1)
        rgb = self.decoder(latent_in)  # [-1, 1]
        rgb = ((rgb.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
        rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()
        if not flat_in:
            rgb = rgb.reshape(B, T, *rgb.shape[1:])
        return rgb

    # ---------- training ----------
    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        """One forward pass with reconstruction + latent-dynamics losses.

        batch keys (all numpy arrays from ReplayBuffer.sample_sequences):
          rgb       (B, L+1, H, W, 3) uint8
          actions   (B, L, 4)         float32

        Loss = MSE reconstruction of all L+1 frames + MSE consistency between
        encoder(rgb_{t+1}) and dynamics(encoder(rgb_t), a_t) for t=0..L-1.
        """
        device = self.device
        rgb_np = batch["rgb"]
        actions_np = batch["actions"]
        x = _rgb_to_tensor(rgb_np, device)  # (B, L+1, 3, H, W)
        a = torch.from_numpy(actions_np).to(device)  # (B, L, 4)

        B, Lp1 = x.shape[:2]
        L = Lp1 - 1
        x_flat = x.reshape(B * Lp1, *x.shape[2:])
        z_flat = self.encoder(x_flat)
        z = z_flat.view(B, Lp1, -1)

        # Reconstruction loss
        recon = self.decoder(z_flat)
        recon_loss = F.mse_loss(recon, x_flat)

        # Dynamics consistency loss: dynamics(z_t, a_t) ≈ z_{t+1} (stop-grad on target)
        z_curr = z[:, :-1].reshape(B * L, -1)
        a_curr = a.reshape(B * L, -1)
        z_pred = self.dynamics(z_curr, a_curr)
        z_target = z[:, 1:].reshape(B * L, -1).detach()
        dyn_loss = F.mse_loss(z_pred, z_target)

        total = recon_loss + dyn_loss
        return {"total": total, "recon": recon_loss.detach(), "dynamics": dyn_loss.detach()}

    # ---------- gradient-tracking helpers used by imagination ----------
    def encode_with_grad(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode RGB tensor *with gradients*. Expects float CHW tensor already."""
        # rgb: (B, T, 3, H, W) or (B, 3, H, W).
        if rgb.dim() == 5:
            B, T = rgb.shape[:2]
            z = self.encoder(rgb.reshape(B * T, *rgb.shape[2:]))
            return z.view(B, T, -1)
        return self.encoder(rgb)

    def step_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """One latent step with gradients. latent (B, D), action (B, A) -> (B, D)."""
        return self.dynamics(latent, action)


class IVideoGPTBackbone(nn.Module):
    """World model that uses iVideoGPT's pretrained tokenizer for encode/decode.

    The tokenizer is a Compressive VQ-VAE pretrained on Open X-Embodiment.
    We use it as a frozen visual encoder/decoder, and learn a continuous
    residual MLP dynamics on top (so the actor-critic gradient path is
    identical to the ``mini`` backbone).

    The latent is the *quantized* codebook embedding, mean-pooled over the
    spatial grid — a (B, latent_dim) continuous vector.

    This is intentionally NOT the full iVideoGPT recipe (which would roll the
    autoregressive transformer in tokens). The transformer path doesn't admit
    a reparameterized policy gradient for the Dreamer actor; the continuous-
    latent setup is the cleaner research starting point. The token-level
    transformer can be wired up in a follow-up if needed.
    """

    def __init__(self, cfg: WMConfig, hf_model_id: str = "thuml/ivideogpt-oxe-64-act-free"):
        super().__init__()
        self.cfg = cfg
        from ivideogpt.vq_model import CompressiveVQModel  # noqa: WPS433 — local import to keep MiniWM CPU-friendly

        self.tokenizer = CompressiveVQModel.from_pretrained(
            hf_model_id, subfolder="tokenizer", low_cpu_mem_usage=False
        )
        # Optionally unfreeze the visual codec for Phase B (variant C). When
        # unfrozen we keep the VQ codebook embedding frozen — iVideoGPT
        # mbrl's `selected_params=True` recipe — so codebook collapse can't
        # masquerade as WM collapse.
        if not cfg.unfreeze_tokenizer:
            for p in self.tokenizer.parameters():
                p.requires_grad_(False)
        else:
            n_trainable = 0
            for name, p in self.tokenizer.named_parameters():
                trainable = not (cfg.freeze_codebook_only and "quantize" in name)
                p.requires_grad_(trainable)
                if trainable:
                    n_trainable += p.numel()
            print(
                f"[IVideoGPTBackbone] tokenizer unfrozen: {n_trainable / 1e6:.2f}M trainable params",
                flush=True,
            )

        # Probe the tokenizer's latent dimension once.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 3, cfg.image_size, cfg.image_size)
            # Use the tokenizer's encoder directly for a continuous-latent forward.
            enc_out = self.tokenizer.encoder(dummy.squeeze(1))
            if enc_out.dim() == 4:
                # (B, C, h, w) -> mean-pooled to (B, C)
                latent_channels = enc_out.shape[1]
            else:
                latent_channels = enc_out.shape[-1]
        self._latent_channels = latent_channels
        self._spatial = enc_out.shape[-1] if enc_out.dim() == 4 else 1

        # Project pooled tokenizer features to our configured latent_dim so
        # the rest of the pipeline doesn't care about the tokenizer's native
        # channel count.
        self.proj_in = nn.Linear(latent_channels, cfg.latent_dim)
        self.proj_out = nn.Linear(cfg.latent_dim, latent_channels * self._spatial * self._spatial)

        self.dynamics = _Dynamics(
            cfg.latent_dim, cfg.action_dim, cfg.dynamics_hidden, cfg.dynamics_layers
        )

    @property
    def latent_dim(self) -> int:
        return self.cfg.latent_dim

    @property
    def device(self) -> torch.device:
        return next(self.dynamics.parameters()).device

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """(B*T, 3, H, W) -> (B*T, latent_dim) via tokenizer encoder + mean pool + proj."""
        if self.cfg.unfreeze_tokenizer:
            feats = self.tokenizer.encoder(x)
        else:
            with torch.no_grad():
                feats = self.tokenizer.encoder(x)
        pooled = feats.flatten(2).mean(-1)
        return self.proj_in(pooled)

    @property
    def encoder(self) -> nn.Module:
        """Alias for ``_encode_frames`` so the pretrain/online code can call
        ``wm.encoder(x)`` symmetrically with the ``mini`` backbone."""
        class _EncWrap(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer
            def forward(self, x):
                return self.outer._encode_frames(x)
        return _EncWrap(self)

    @torch.no_grad()
    def encode_frame(self, rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = _rgb_to_tensor(rgb, self.device)  # (B, T, 3, H, W)
        B, T = x.shape[:2]
        z = self._encode_frames(x.reshape(B * T, *x.shape[2:]))
        return z.view(B, T, -1)[:, -1]

    @torch.no_grad()
    def rollout_latent(self, init_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        zs = [init_latent]
        z = init_latent
        for h in range(actions.shape[1]):
            z = self.dynamics(z, actions[:, h])
            zs.append(z)
        return torch.stack(zs, dim=1)

    @torch.no_grad()
    def decode_to_rgb(self, latent: torch.Tensor) -> np.ndarray:
        """Project pooled latent back through tokenizer.decoder (best-effort).

        The pre-quant pooled latent doesn't carry full spatial info, so this
        reconstruction is intentionally coarse; only for pixel-metric
        evaluation, not training.
        """
        flat_in = latent.dim() == 2
        latent_in = latent if flat_in else latent.reshape(-1, latent.shape[-1])
        feats = self.proj_out(latent_in).view(
            -1, self._latent_channels, self._spatial, self._spatial
        )
        rgb = self.tokenizer.decoder(feats)  # (B, 3, H, W) in [0,1]
        rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return rgb

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        """Train dynamics + projection layers; tokenizer trains iff unfrozen.

        When the tokenizer is frozen we regress feats_pred -> feats_target
        in the codec's feature space (target is the no-grad encoder output).
        When unfrozen we regress decoded pixels against the raw image, which
        gives the encoder a real gradient signal — analogous to iVideoGPT's
        L1+LPIPS pixel reconstruction (we drop LPIPS here to avoid pulling
        the perceptual model on Machine B's tight render budget).
        """
        device = self.device
        rgb_np = batch["rgb"]
        actions_np = batch["actions"]
        x = _rgb_to_tensor(rgb_np, device)  # in [-1, 1]
        a = torch.from_numpy(actions_np).to(device)
        B, Lp1 = x.shape[:2]
        L = Lp1 - 1
        x_flat = x.reshape(B * Lp1, *x.shape[2:])
        z_flat = self._encode_frames(x_flat)
        z = z_flat.view(B, Lp1, -1)

        feats_pred = self.proj_out(z_flat).view(
            -1, self._latent_channels, self._spatial, self._spatial
        )

        if self.cfg.unfreeze_tokenizer:
            # Pixel-space reconstruction. Tokenizer.decoder is trained
            # together with the encoder; codebook stays frozen.
            recon_pixels = self.tokenizer.decoder(feats_pred)  # decoder output in [0, 1]
            x_pixel = (x_flat + 1.0) * 0.5  # convert [-1, 1] -> [0, 1] for comparison
            recon_loss = F.mse_loss(recon_pixels, x_pixel)
        else:
            with torch.no_grad():
                feats_target = self.tokenizer.encoder(x_flat)  # (B*Lp1, C, h, w)
            recon_loss = F.mse_loss(feats_pred, feats_target)

        # Dynamics consistency.
        z_curr = z[:, :-1].reshape(B * L, -1)
        a_curr = a.reshape(B * L, -1)
        z_pred = self.dynamics(z_curr, a_curr)
        z_target = z[:, 1:].reshape(B * L, -1).detach()
        dyn_loss = F.mse_loss(z_pred, z_target)

        total = recon_loss + dyn_loss
        return {"total": total, "recon": recon_loss.detach(), "dynamics": dyn_loss.detach()}

    def encode_with_grad(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.dim() == 5:
            B, T = rgb.shape[:2]
            z = self._encode_frames(rgb.reshape(B * T, *rgb.shape[2:]))
            return z.view(B, T, -1)
        return self._encode_frames(rgb)

    def step_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(latent, action)


def build_world_model(cfg: WMConfig) -> nn.Module:
    if cfg.backbone == "ivideogpt":
        return IVideoGPTBackbone(cfg)
    return MiniWorldModel(cfg)
