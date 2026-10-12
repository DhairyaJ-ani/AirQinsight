"""Loss functions for the spatiotemporal NO₂ forecasting model.

Combined loss:
    L = w_mse  · L_MSE
      + w_grad · L_gradient
      + w_mass · L_mass

L_MSE        : mean squared error on log1p-normalised NO₂.
L_gradient   : penalises blurring by matching spatial gradients of the prediction.
L_mass       : enforces mass conservation across the forecast horizon.

Mathematical formulation
------------------------
Gradient loss (Sobel approximation):
    G_x = ∂C/∂x   G_y = ∂C/∂y
    L_grad = MSE(G_x_pred, G_x_true) + MSE(G_y_pred, G_y_true)

Mass conservation loss:
    m̂_k = Σ_ij ŷ_k(i,j)   (predicted total)
    m_k  = Σ_ij  y_k(i,j)  (target total)
    L_mass = mean_k [ (m̂_k - m_k)² / (m_k + ε)² ]

All losses require PyTorch and are designed to work with torch.cuda.amp
automatic mixed precision.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Gradient (Sobel) operators – registered as buffers once
# ---------------------------------------------------------------------------

def _sobel_kernels():
    """Return (Kx, Ky) Sobel kernels as (1,1,3,3) float tensors."""
    import torch
    kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                      dtype=torch.float32).unsqueeze(0)
    ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                      dtype=torch.float32).unsqueeze(0)
    return kx, ky


def _spatial_gradients(x):
    """Compute Sobel gradients for each (B, C, H, W) tensor using F.conv2d."""
    import torch
    import torch.nn.functional as F
    kx, ky = _sobel_kernels()
    kx = kx.to(x.device)
    ky = ky.to(x.device)
    B, C, H, W = x.shape
    x_flat = x.reshape(B * C, 1, H, W)
    gx = F.conv2d(x_flat, kx, padding=1)
    gy = F.conv2d(x_flat, ky, padding=1)
    return gx.reshape(B, C, H, W), gy.reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def mse_loss(pred, target):
    """Standard MSE."""
    import torch.nn.functional as F
    return F.mse_loss(pred, target)


def gradient_loss(pred, target):
    """Gradient (sharpness) loss via Sobel operators."""
    p_gx, p_gy = _spatial_gradients(pred)
    t_gx, t_gy = _spatial_gradients(target)
    import torch.nn.functional as F
    return 0.5 * (F.mse_loss(p_gx, t_gx) + F.mse_loss(p_gy, t_gy))


def mass_conservation_loss(pred, target):
    """Relative mass conservation loss across forecast horizons.

    pred, target : (B, horizons, H, W)
    """
    import torch
    # Sum over spatial dims
    m_pred = pred.sum(dim=(-1, -2))    # (B, horizons)
    m_true = target.sum(dim=(-1, -2))  # (B, horizons)
    eps = 1.0
    loss = ((m_pred - m_true) ** 2 / (m_true + eps) ** 2)
    return loss.mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class ForecastLoss:
    """Weighted combination of MSE + gradient + mass conservation losses.

    Parameters
    ----------
    w_mse, w_grad, w_mass : loss weights (floats)
    """

    def __init__(
        self,
        w_mse: float = 1.0,
        w_grad: float = 0.3,
        w_mass: float = 0.2,
    ):
        if not _torch_available():
            raise RuntimeError("PyTorch is required for ForecastLoss")
        self.w_mse = w_mse
        self.w_grad = w_grad
        self.w_mass = w_mass

    def __call__(self, pred, target) -> dict:
        """Compute losses and return a dict of individual and total values.

        Parameters
        ----------
        pred   : (B, horizons, H, W) predicted residual + physics
        target : (B, horizons, H, W) true future NO₂

        Returns
        -------
        dict with keys: total, mse, gradient, mass
        """
        l_mse  = mse_loss(pred, target)
        l_grad = gradient_loss(pred, target)
        l_mass = mass_conservation_loss(pred, target)
        total  = self.w_mse * l_mse + self.w_grad * l_grad + self.w_mass * l_mass
        return {
            "total":    total,
            "mse":      l_mse.detach(),
            "gradient": l_grad.detach(),
            "mass":     l_mass.detach(),
        }

    def numpy_eval(
        self,
        pred: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, float]:
        """CPU-only evaluation for validation without autograd."""
        import torch
        with torch.no_grad():
            p = torch.from_numpy(pred.astype(np.float32))
            t = torch.from_numpy(target.astype(np.float32))
            losses = self(p, t)
        return {k: float(v) for k, v in losses.items()}


# ---------------------------------------------------------------------------
# Mask-aware loss (excludes cloud-contaminated / gap-filled pixels)
# ---------------------------------------------------------------------------

class MaskAwareForecastLoss:
    """Weighted MSE + gradient + mass conservation with a valid-pixel cloud mask.

    Cloud-covered or gap-filled pixels are excluded from MSE and gradient loss
    terms, preventing the training signal from being biased towards zero in
    cloudy regions.

    Parameters
    ----------
    w_mse, w_grad, w_mass : loss weights (floats).
    min_valid_frac : minimum fraction of valid (unmasked) pixels required
        across a batch to compute a meaningful loss.  If fewer pixels are
        valid the loss falls back to all-pixel MSE.

    Usage
    -----
        loss_fn = MaskAwareForecastLoss()
        # valid_mask: (B, horizons, H, W) bool, True = observed/valid pixel
        losses = loss_fn(pred, target, valid_mask)
    """

    def __init__(
        self,
        w_mse: float = 1.0,
        w_grad: float = 0.3,
        w_mass: float = 0.2,
        min_valid_frac: float = 0.05,
    ):
        if not _torch_available():
            raise RuntimeError("PyTorch is required for MaskAwareForecastLoss")
        self.w_mse = w_mse
        self.w_grad = w_grad
        self.w_mass = w_mass
        self.min_valid_frac = min_valid_frac

    def _masked_mse(self, pred, target, mask):
        """MSE computed only over valid pixels."""
        diff = (pred - target) ** 2
        valid = mask.float()
        total_valid = valid.sum()
        if float(total_valid.item()) / float(valid.numel()) < self.min_valid_frac:
            # Not enough valid pixels; fall back to unmasked MSE
            return diff.mean()
        return (diff * valid).sum() / (total_valid + 1e-8)

    def _masked_gradient_loss(self, pred, target, mask):
        """Gradient loss restricted to valid pixels."""
        import torch
        import torch.nn.functional as F
        p_gx, p_gy = _spatial_gradients(pred)
        t_gx, t_gy = _spatial_gradients(target)
        # The gradient at a pixel is valid only when that pixel and its
        # neighbours are valid.  Approximate by dilating the mask inward
        # with a 3x3 max-pool of the inverted mask (any masked neighbour
        # invalidates the gradient).
        B, C, H, W = mask.shape
        m_flat = mask.float().reshape(B * C, 1, H, W)
        invalid_g = F.max_pool2d(1.0 - m_flat, 3, stride=1, padding=1)
        valid_g = (1.0 - invalid_g).reshape(B, C, H, W)
        n_valid = valid_g.sum()
        if float(n_valid.item()) < 1.0:
            return torch.tensor(0.0, device=pred.device)
        loss_x = ((p_gx - t_gx) ** 2 * valid_g).sum() / (n_valid + 1e-8)
        loss_y = ((p_gy - t_gy) ** 2 * valid_g).sum() / (n_valid + 1e-8)
        return 0.5 * (loss_x + loss_y)

    def __call__(self, pred, target, valid_mask=None) -> dict:
        """Compute masked losses.

        Parameters
        ----------
        pred       : (B, horizons, H, W)
        target     : (B, horizons, H, W)
        valid_mask : (B, horizons, H, W) bool tensor, True = valid/observed pixel.
                     If None, falls back to standard unmasked loss.

        Returns
        -------
        dict with keys: total, mse, gradient, mass, valid_fraction
        """
        import torch
        if valid_mask is None:
            valid_mask = torch.ones_like(pred, dtype=torch.bool)
        mask = valid_mask.to(pred.device)
        valid_frac = float(mask.float().mean().item())

        l_mse  = self._masked_mse(pred, target, mask)
        l_grad = self._masked_gradient_loss(pred, target, mask)
        l_mass = mass_conservation_loss(pred, target)  # mass uses all pixels
        total  = self.w_mse * l_mse + self.w_grad * l_grad + self.w_mass * l_mass
        return {
            "total":          total,
            "mse":            l_mse.detach(),
            "gradient":       l_grad.detach(),
            "mass":           l_mass.detach(),
            "valid_fraction": valid_frac,
        }

    def numpy_eval(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        """CPU-only evaluation without autograd."""
        import torch
        with torch.no_grad():
            p = torch.from_numpy(pred.astype(np.float32))
            t = torch.from_numpy(target.astype(np.float32))
            m = (torch.from_numpy(valid_mask.astype(bool))
                 if valid_mask is not None else None)
            losses = self(p, t, m)
        return {k: float(v) for k, v in losses.items()}
