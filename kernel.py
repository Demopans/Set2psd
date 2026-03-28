try:
    import cupy as cp
except (ModuleNotFoundError, ImportError) as e:
    print(e)
try:
    import dpnp as dp
except (ModuleNotFoundError, ImportError) as e:
    print(e)

import numpy as np

compositeImage = np.ndarray[tuple[int,int], np.uint32]

# kernels have consistent API
class Kernel:
    @staticmethod
    def main(root: compositeImage, b: list[compositeImage]) -> list[compositeImage]: pass

class GPUKernel(Kernel):
    # custom kernel
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
            case cp.uint32: GPUKernel.transparentMask32(a, b, r)
            case cp.uint64: GPUKernel.transparentMask64(a, b, r)
        return r

    @staticmethod
    def main(root: compositeImage, b: list[compositeImage]) -> list[compositeImage]:
        return [GPUKernel.operate(root, _) for _ in b]

class CPUKernel(Kernel):
    @staticmethod
    def main(root: compositeImage, b: list[compositeImage]) -> list[compositeImage]:
        return [np.where(root == _, 0, _) for _ in b]

class IntelKernel(Kernel):
    @staticmethod
    def main(root: compositeImage, b: list[compositeImage]) -> list[compositeImage]:
        return [dpnp.where(root == _, 0, _) for _ in b]