"""Differentiable linear prediction: STFT-domain filtering, Levinson recursions,
log-area-ratio parametrisation and formant picking from the all-pole polynomial."""
import torch
import torch.nn.functional as F


def ceil_division(n: int, d: int) -> int:
    return -(n // -d)


class LFilter(torch.nn.Module):
    """Time-varying FIR/all-pole filtering in the STFT domain (windowed overlap-add).

    Filter coefficients are given per frame, shape (batch, n_coeffs, n_frames).
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length).reshape(1, -1, 1), persistent=False)
        self.scale = 2.0 * hop_length / win_length
        self.fold_kwargs = dict(kernel_size=(win_length, 1), stride=(hop_length, 1), padding=0)

    @staticmethod
    def _pad_frames(frames, target_len):
        n_pad = target_len - frames.size(-1)
        return F.pad(frames, pad=(n_pad // 2, ceil_division(n_pad, 2)), mode="replicate")

    def forward(self, x, b=None, a=None):
        """x: (batch, 1, time); b, a: (batch, n_coeffs, n_frames). Returns (batch, 1, time)."""
        if x.size(1) != 1:
            raise RuntimeError(f"channels must be 1, got shape {x.shape}")
        num_frames = ceil_division(x.size(-1), self.hop_length)
        left_pad = self.win_length
        right_pad = num_frames * self.hop_length - x.size(-1) + self.win_length
        x_padded = F.pad(x, pad=(left_pad, right_pad)).unsqueeze(-1)
        fold_size = x_padded.shape[2:]

        x_windowed = F.unfold(x_padded, **self.fold_kwargs) * self.window
        X = torch.fft.rfft(x_windowed, n=self.n_fft, dim=1)
        A = torch.ones_like(X) if a is None else torch.fft.rfft(self._pad_frames(a, X.size(-1)), n=self.n_fft, dim=1)
        B = torch.ones_like(X) if b is None else torch.fft.rfft(self._pad_frames(b, X.size(-1)), n=self.n_fft, dim=1)

        y_windowed = torch.fft.irfft(X * B / A, n=self.n_fft, dim=1)[:, :self.win_length, :]
        y = F.fold(y_windowed, output_size=fold_size, **self.fold_kwargs)
        return y[:, :, left_pad:-right_pad, 0] * self.scale


class Emphasis(torch.nn.Module):
    """Pre-emphasis 1 - alpha z^-1 (applied with its own STFT filter, hop 256)."""

    def __init__(self, alpha=0.97):
        super().__init__()
        self.lfilter = LFilter(n_fft=512, hop_length=256, win_length=512)
        self.register_buffer("coeffs", torch.tensor([1, -alpha], dtype=torch.float32))

    def forward(self, x):
        n_frames = ceil_division(x.size(-1), self.lfilter.hop_length)
        b = self.coeffs.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, n_frames).to(x.device)
        return self.lfilter(x, b=b, a=None)


def levinson(R: torch.Tensor, M: int, eps: float = 1e-3) -> torch.Tensor:
    """Autocorrelation (..., >=M+1) -> all-pole polynomial (..., M+1), a[..., 0] = 1."""
    R = R / R[..., 0:1]
    R[..., 0] = R[..., 0] + eps  # white noise correction
    K = torch.sum(R[..., 1:2], dim=-1, keepdim=True)
    A = torch.cat([-1.0 * K, torch.ones_like(R[..., 0:1])], dim=-1)
    E = 1.0 - K ** 2
    for p in torch.arange(1, M, device=R.device):
        K = torch.sum(A[..., 0:p + 1] * R[..., 1:p + 2], dim=-1, keepdim=True) / E
        if K.abs().max() > 1.0:
            raise ValueError(f"Unstable filter, |K| was {K.abs().max()}")
        A = torch.cat([-1.0 * K,
                       A[..., 0:p] - 1.0 * K * torch.flip(A[..., 0:p], dims=[-1]),
                       torch.ones_like(R[..., 0:1])], dim=-1)
        E = E * (1.0 - K ** 2)
    return torch.flip(A, dims=[-1])


def forward_levinson(K: torch.Tensor) -> torch.Tensor:
    """Reflection coefficients (..., M) -> all-pole polynomial (..., M+1), a[..., 0] = 1."""
    M = K.size(-1)
    A = -1.0 * K[..., 0:1]
    for p in torch.arange(1, M, device=K.device):
        A = torch.cat([-1.0 * K[..., p:p + 1],
                       A[..., 0:p] - 1.0 * K[..., p:p + 1] * torch.flip(A[..., 0:p], dims=[-1])], dim=-1)
    A = torch.cat([A, torch.ones_like(A[..., 0:1])], dim=-1)
    return torch.flip(A, dims=[-1])


def lar_to_poly(lars: torch.Tensor) -> torch.Tensor:
    """Log-area ratios (batch, P, frames) -> polynomial (batch, P+1, frames).

    k = tanh(LAR / 2) keeps |k| < 1, so the synthesis filter is always stable.
    """
    k = torch.tanh(0.5 * lars)
    return forward_levinson(k.permute(0, 2, 1)).permute(0, 2, 1)


class LinearPredictor(torch.nn.Module):
    """Autocorrelation-method LP per STFT frame, plus inverse filtering."""

    def __init__(self, n_fft=512, hop_length=160, win_length=512, order=16):
        super().__init__()
        self.n_fft, self.hop_length, self.win_length, self.order = n_fft, hop_length, win_length, order
        self.lfilter = LFilter(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.register_buffer("window", torch.hann_window(win_length))

    def estimate(self, x: torch.Tensor):
        """x: (batch, time) -> a: (batch, order+1, frames), gain g: (batch, frames, 1)."""
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                       return_complex=True, window=self.window.to(x.device))
        power = (torch.abs(X) ** 2).transpose(1, 2)
        r = torch.fft.irfft(power, dim=-1)
        r[..., 0] = r[..., 0] + 1e-6
        a = levinson(r, self.order)
        g = torch.sqrt(torch.sum(r[..., :self.order + 1] * a, dim=-1, keepdim=True))
        return a.transpose(1, 2), g

    def inverse_filter(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Prediction residual e = A(z) x. x: (batch, 1, time), a: (batch, order+1, frames)."""
        return self.lfilter(x=x, b=a, a=None)


def get_formants(a: torch.Tensor, fs: int = 16000) -> torch.Tensor:
    """Formants from the roots of A(z) for the first item in the batch.

    a: (batch, order+1, frames). Returns (frames, 4) in Hz: the four lowest roots
    with 90 < f < 5500 Hz and bandwidth < 400 Hz, zero-padded if fewer are found.
    """
    coeffs = a[0].detach().cpu()
    n, num_frames = coeffs.shape
    companion = torch.zeros((num_frames, n - 1, n - 1))
    companion[:, 1:, :-1] = torch.eye(n - 2).unsqueeze(0).repeat(num_frames, 1, 1)
    companion[:, 0, :] = -coeffs[1:, :].transpose(0, 1)
    roots = torch.linalg.eigvals(companion)
    roots = roots * (roots.imag >= 0)

    freqs = torch.atan2(roots.imag, roots.real) * (fs / (2 * torch.pi))
    bandwidths = -1 / 2 * (fs / (2 * torch.pi)) * torch.log(torch.abs(roots))
    valid = (freqs > 90) & (freqs < 5500) & (bandwidths < 400)
    freqs = freqs * valid
    freqs[~valid] = float("inf")
    freqs, _ = torch.sort(freqs, dim=1)
    freqs = freqs[:, :4]
    freqs = torch.where(freqs == float("inf"), torch.tensor(0.0), freqs)
    if freqs.size(1) < 4:
        freqs = F.pad(freqs, (0, 4 - freqs.size(1)), value=0.0)
    return freqs


def envelope_db(a: torch.Tensor, n_fft: int = 512) -> torch.Tensor:
    """Magnitude of 1/A(e^jw) in dB for the first item. a: (batch, order+1, frames) -> (n_fft//2+1, frames)."""
    return -20 * torch.log10(torch.fft.rfft(a[0].detach().cpu(), n=n_fft, dim=0).abs() + 1e-8)
