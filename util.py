"""
util.py

AUTO-GENERATED DOCUMENTATION (Grok-assisted)
=============================================

Core utility library for the differential PNG compressor.

Key concepts:
- Flattening: converts 3-channel/4-channel images (H×W×4 uint8) into a 2D packed
  compositeImage (H×W uint32/uint64) so the kernel can operate on full ARGB pixels
  in a single comparison.
- Tree-based batching: builds a hierarchical dependency tree so each image only
  stores deltas against its immediate ancestor(s). This dramatically reduces
  redundancy across large sets of similar images (e.g. video frames, UI variants).
- Async I/O via ThreadPoolExecutor + 7z packing for maximum speed and compression.

The ROUTER global selects the fastest available kernel at import time:
    1. CuPy GPU (if CUDA available)
    2. Intel dpnp (if oneAPI devices present)
    3. NumPy CPU fallback
"""

import concurrent.futures as cfutures, os, cv2, PIL.Image as PILI, numpy as np, py7zr
from math import ceil

from kernel import CPUKernel, GPUKernel, IntelKernel

ROUTER = CPUKernel
try:
    import cupy as xp

    if xp.cuda.is_available():
        print("🚀 CuPy GPU detected – using accelerated kernels")
        _USE_GPU = True
        ROUTER = GPUKernel
    else:
        raise ImportError("CuPy not available, only use_gpu=False will work")
except (ModuleNotFoundError, ImportError) as e:
    print(e)
    print('Failed to import CuPy. Falling back to CPU')
    _USE_GPU = False
    try:
        import dpctl
        if len(dpctl.get_devices())==0: raise ImportError("No Intel devices detected")
        import dpnp as xp
        ROUTER = IntelKernel
    except (ModuleNotFoundError, ImportError) as e:
        print(e)
        print('Failed to import Intel Data Parallel Extension for NumPy. Falling back to CPU')
        import numpy as xp

bite = { 1: xp.uint8, 2: xp.uint16, 4: xp.uint32, 8: xp.uint64 }
cDepth, bitDepth = bite[2], bite[8]
syspath = list[str]
imgshape = tuple[int, int]
image = xp.ndarray[tuple[int, int, int], cDepth]  # explicit cross CPU GPU compat
executor = cfutures.ThreadPoolExecutor(max_workers=os.cpu_count() // 2)  # prefer physical cores


class Util:
    """
    Namespace for all static helper utilities.
    """
    class IO:
        """I/O helpers – all heavy calls are wrapped in futures for async scheduling."""
        @staticmethod
        def getImageInfo(path: str) -> tuple[str | None, object, imgshape]:
            with PILI.open(path) as img:
                w, h = img.size
                f = img.format
                s = xp.array(img).dtype
                return f, s, (w, h)

        # asyncs get constant structure, even if boilerplate
        @staticmethod
        def readPNG(root: str, file: str)-> cfutures.Future:
            def _readPNG(root: str, file: str) -> image:
                return xp.asarray(cv2.imread(f'{root}/{file}', cv2.IMREAD_UNCHANGED))
            return executor.submit(_readPNG, root, file)

        @staticmethod
        def writePNG(root: str, file: str, img: image) -> cfutures.Future:
            def _writePNG(root: str, file: str, img):
                    match type(img):
                        case np.ndarray: cv2.imwrite(f'{root}/_{file}', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                        case xp.ndarray: cv2.imwrite(f'{root}/_{file}', img.get(), [cv2.IMWRITE_PNG_COMPRESSION, 9])
            return executor.submit(_writePNG, root, file, img)

        @staticmethod
        def writeMeta(root: str, files: list[str], meta: list[list[int]]) -> cfutures.Future:
            def _writeMeta(root: str, files: list[str], meta: list[list[int]]):
                with open(f'{root}/info.txt', 'w') as f:
                    for i, file in zip(meta, files): f.write(f'"{file}," {str(i).strip('[]') + '\n'}')
            return executor.submit(_writeMeta, root, files, meta)

    mode2bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32,
                "F": 32, "I;16": 16, "I;16B": 16, "I;16L": 16, "I;16S": 16, "I;16BS": 16, "I;16LS": 16, "I;32": 32,
                "I;32B": 32, "I;32L": 32, "I;32S": 32, "I;32BS": 32, "I;32LS": 32}

    # ─────────────────────────────────────────────────────────────────────────
    # Flatten / Fatten – convert between packed 2D compositeImage and 3D image
    # ─────────────────────────────────────────────────────────────────────────
    flatten = lambda a, dtypes: a.view(dtype=bite[dtypes.itemsize * 4]).reshape(a.shape[:2])
    fatten = lambda a, dtypes: a.view(dtype=bite[dtypes.itemsize // 4]).reshape((a.shape[0],a.shape[1], 4))

    @staticmethod
    def makeTree(files: list[str], interval) -> list[list[int]]:
        out: list[list[int]] = []
        for i, v in enumerate(files):
            app: list[int]
            s = i % interval
            match s:
                # subtree root
                case 0 if i == 0:
                    app = [0]
                case 0:
                    app = [0, i]
                # subtree
                case _ if i // interval == 0:
                    app = [0, i]
                case _:
                    app = [0, i // interval * interval, i]
            out.append(app)
        return out

    class Compress:
        method = {
            'lzma': py7zr.FILTER_LZMA,
            'lzma2': py7zr.FILTER_LZMA2,
            'bzip': py7zr.FILTER_BZIP2,
            'deflate': py7zr.FILTER_DEFLATE,
            'store': py7zr.FILTER_COPY
        }
        ext = {
            'lzma': '7z',
            'lzma2': '7z',
            'bzip': 'bz2',
            'deflate': 'zip',
            'store': 'zip'
        }
        @staticmethod
        def wipe(root: str, name: str, method: str):
            py7zr.SevenZipFile(f"{root}/{name}.{Util.Compress.ext[method]}", 'w', filters=[{"id": Util.Compress.method[method], "preset": 7}])

        @staticmethod
        def compress(root: str, files: list[str], name: str, method: str):
            with py7zr.SevenZipFile(f"{root}/{name}.{Util.Compress.ext[method]}", 'a', filters=[{"id": Util.Compress.method[method] , "preset": 7}]) as archive:
                archive.write(f'{root}/info.txt', 'info.txt')
                for file in files:
                    archive.write(f'{root}/_{file}', file)

class Runner:
    @staticmethod
    def runner(root: str, files: list[str]) -> list[image]:
        # load images
        ima = [Util.IO.readPNG(root, _) for _ in files]
        ima = [Util.flatten(a := _.result(), a.dtype) for _ in ima]
        rt, br = ima[0], ima[1:]
        # run comp
        br = ROUTER.main(rt, br)
        # return imgs
        return [Util.fatten(_,_.dtype) for _ in br]

    @staticmethod
    def main(root: str, files: list[str], batchSize: int, name: str, compress: str = 'lzma2'):
        """
        Full compression workflow:

        1. Validate all images are identical PNGs (size + bit depth)
        2. Build dependency tree
        3. Process internal batch members (deltas vs batch root)
        4. Process batch roots themselves (deltas vs global root 0)
        5. Write reconstruction metadata
        6. 7z-pack everything
        7. Clean up temporary _files
        """
        import operator
        # verify
        if len(meta := set([Util.IO.getImageInfo(f'{root}/{_}') for _ in files])) != 1:
            exit("inconsistent size/bit depth")
        elif (meta := list(meta)[0])[0] != 'PNG':
            exit("not PNG")

        # generate tree
        tree = Util.makeTree(files, batchSize)
        batches = [
            tree[i * batchSize:min(len(files), i * batchSize + batchSize)] for i in range(ceil(len(files) / batchSize))
        ]
        # feed GPU loop by batch
        for b in batches:
            f = files[b[0][-1]:b[-1][-1]+1]
            res = Runner.runner(root, f)
            # schedule write before next loop
            [Util.IO.writePNG(root,file,_) for file,_ in zip(f[1:],res)]

        # proc the subroots
        if len(batches) > 1 :
            f = operator.itemgetter(*[_[0][-1] for _ in batches])(files)
            res = Runner.runner(root, f)

        batches, b = [Util.IO.writePNG(root, file, _) for file, _ in zip(f[1:], res)], Util.IO.writePNG(root,files[0],Util.IO.readPNG(root,files[0]).result())
        # write metadata
        Util.IO.writeMeta(root, files, tree).result()
        [_.result() for _ in batches]
        b.result()
        # pack
        Util.Compress.wipe(root, name, compress)
        Util.Compress.compress(root, files, name, compress)
        # rm leftover files
        [os.remove(f'{root}/_{_}') for _ in files]
        os.remove(f'{root}/info.txt')

class CLI:
    @staticmethod
    def main():
        pass