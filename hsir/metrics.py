import functools
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Global format setting
FORMAT_HWC = 'hwc'
FORMAT_CHW = 'chw'
DATA_FORMAT = FORMAT_CHW  # Default format

def set_data_format(format):
    """Set the global data format (either 'chw' or 'hwc')"""
    global DATA_FORMAT
    if format.lower() not in {FORMAT_HWC, FORMAT_CHW}:
        raise ValueError('Invalid data format, choose from "HWC" or "CHW"')
    DATA_FORMAT = format.lower()

def CHW2HWC(func):
    """Decorator to convert data format from CHW to HWC."""
    @functools.wraps(func)
    def wrapper(output, target, *args, **kwargs):
        if DATA_FORMAT == FORMAT_CHW:
            output = output.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)
        return func(output, target, *args, **kwargs)
    return wrapper

def torch2numpy(func):
    """Decorator to convert torch tensors to numpy arrays."""
    @functools.wraps(func)
    def wrapper(output, target, *args, **kwargs):
        if isinstance(output, np.ndarray):
            return func(output, target, *args, **kwargs)
        return func(output.detach().cpu().numpy(), target.detach().cpu().numpy(), *args, **kwargs)
    return wrapper

def enable_batch_input(reduce=True):
    """Decorator to enable batch processing."""
    def inner(func):
        @functools.wraps(func)
        def wrapper(output, target, *args, **kwargs):
            if len(output.shape) == 4:
                b = output.shape[0]
                results = [func(output[i], target[i], *args, **kwargs) for i in range(b)]
                return np.mean(results) if reduce else results
            return func(output, target, *args, **kwargs)
        return wrapper
    return inner

def bandwise(func):
    """Decorator to apply function channel-wise (for multi-band images)."""
    @functools.wraps(func)
    def wrapper(output, target, *args, **kwargs):
        channels = output.shape[-3] if DATA_FORMAT == FORMAT_CHW else output.shape[-1]
        total = sum(func(output[..., ch], target[..., ch], *args, **kwargs) for ch in range(channels))
        return total / channels
    return wrapper

# Metric Functions
@torch2numpy
@enable_batch_input()
@CHW2HWC
def sam(img1, img2, eps=1e-8):
    """
    Spectral Angle Mapper (SAM), which computes the spectral similarity
    between two spectra.
    """
    tmp1 = np.sum(img1 * img2, axis=2) + eps
    tmp2 = np.sqrt(np.sum(img1**2, axis=2)) + eps
    tmp3 = np.sqrt(np.sum(img2**2, axis=2)) + eps
    tmp4 = tmp1 / (tmp2 * tmp3)
    angle = np.arccos(tmp4.clip(-1, 1))
    return np.mean(np.real(angle))

@torch2numpy
@enable_batch_input()
@bandwise
def mpsnr(output, target, data_range=1):
    """Compute the Mean PSNR (Peak Signal-to-Noise Ratio)."""
    return peak_signal_noise_ratio(target, output, data_range=data_range)

@torch2numpy
@enable_batch_input()
@bandwise
def mssim(img1, img2, **kwargs):
    """Compute the Mean SSIM (Structural Similarity Index)."""
    return structural_similarity(img1, img2, data_range=1)

# @torch2numpy
# @enable_batch_input()
# @bandwise
# def mssim(img1, img2, **kwargs):
#     """Compute the Mean SSIM (Structural Similarity Index)."""
#     return structural_similarity(img1, img2, **kwargs)