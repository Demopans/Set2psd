import cupy as cp
import numpy as np

compositeImage = np.ndarray[tuple[int,int], np.uint32]

# kernels have consistent API
class GPUKernel:
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

class CPUKernel:
    # numpy operator
    transparentMask32 = lambda a,b: 0x0000_0000 if a==b else b
    transparentMask64 = lambda a,b: 0x0000_0000_0000_0000 if a==b else b

    @staticmethod
    def operate(a: compositeImage, b: compositeImage) -> compositeImage:
        match a.dtype:
            case np.uint32: opt = np.vectorize(CPUKernel.transparentMask32)
            case np.uint64: opt = np.vectorize(CPUKernel.transparentMask64)
        return opt(a, b)

    @staticmethod
    def main(root: compositeImage, b: list[compositeImage]) -> list[compositeImage]:
        return [CPUKernel.operate(root, _) for _ in b]
