"""
kernel.py

AUTO-GENERATED DOCUMENTATION (Grok-assisted)
=============================================

This module defines the core differential kernel operations for the Puffo image compressor.
It implements a pixel-wise "transparent mask" that turns identical pixels between a root image
and a branch image into fully transparent (0x00000000 / 0x0000000000000000). This enables
highly efficient delta compression: unchanged areas become transparent, allowing 7z (or other)
to collapse them to near-zero size.

The design supports three backends:
- GPU (CuPy) – fastest for large batches
- Intel oneAPI (dpnp) – accelerated on Intel GPUs/CPU
- Pure NumPy (CPU fallback)

All kernels share the same public API for easy swapping via the ROUTER in util.py.
"""

# Type alias for the flattened composite image used by all kernels
# (height × width, uint32 or uint64 representing packed ARGB)
compositeImage: type

try:
    import cupy as cp

    compositeImage = cp.ndarray[tuple[int, int], cp.uint32 | cp.uint64]
except (ModuleNotFoundError, ImportError) as e:
    print(e)
    try:
        import dpnp as dp

        compositeImage = dp.ndarray[tuple[int, int], dp.uint32 | dp.uint64]
    except (ModuleNotFoundError, ImportError) as e:
        print(e)
        import numpy as np

        compositeImage = np.ndarray[tuple[int, int], np.uint32 | np.uint64]


# kernels have consistent API
class Kernel:
    """
    Abstract base class defining the kernel contract.

    Every concrete kernel must implement:
        main(root: compositeImage, b: list[compositeImage]) -> list[compositeImage]
    """

    @staticmethod
    def main(root: compositeImage, b: list[compositeImage]) -> list[compositeImage]: pass


class GPUKernel(Kernel):
    """
    CuPy GPU kernel – uses custom ElementwiseKernel for maximum performance.

    The kernel performs a single-pass pixel comparison:
        if x == y: out = 0 (transparent)
        else:      out = y (keep changed pixel)

    This preserves the original ARGB values of differing pixels while zeroing
    everything that matches the root, enabling perfect reconstruction via compositing.
    """

    # Pre-compiled CuPy element-wise kernels (one for 32-bit, one for 64-bit packing)
    transparentMask32 = cp.ElementwiseKernel(
        'uint32 x, uint32 y',
        'uint32 out',
        '''
        out = x == y ? 0x00000000u & x : y; // ARGB spec, blanks out bytes if equal, keeps y if not (designed to retain transparency for image reconstruction)
        ''',
        'transparentMask'
    )
    transparentMask64 = cp.ElementwiseKernel(
        'uint64 x, uint64 y',
        'uint64 out',
        '''
        out = x == y ? 0x0000000000000000u & x : y; // ARGB spec, blanks out bytes if equal, keeps y if not (designed to retain transparency for image reconstruction)
        ''',
        'transparentMask'
    )

    @staticmethod
    def operate(a: compositeImage, b: compositeImage) -> compositeImage:
        r = cp.empty_like(a)
        match a.dtype:
            case cp.uint32:
                GPUKernel.transparentMask32(a, b, r)
            case cp.uint64:
                GPUKernel.transparentMask64(a, b, r)
        return r

    @staticmethod
    def main(root: compositeImage, b: list[compositeImage]) -> list[compositeImage]:
        return [GPUKernel.operate(root, _) for _ in b]


class CPUKernel(Kernel):
    """
    Pure NumPy CPU fallback – uses np.where for the same transparent-mask logic.
    Slower than GPU but guaranteed to work everywhere.
    """
    @staticmethod
    def main(root: compositeImage, b: list[compositeImage]) -> list[compositeImage]:
        return [np.where(root == _, 0, _) for _ in b]


class IntelKernel(Kernel):
    """
    Intel oneAPI (dpnp) kernel – uses dpnp.where for acceleration on Intel hardware.
    """
    @staticmethod
    def main(root: compositeImage, b: list[compositeImage]) -> list[compositeImage]:
        return [dpnp.where(root == _, 0, _) for _ in b]
