import asyncio, concurrent.futures as cfutures, numpy as ni, os

# GPU / CPU fallback
try:
    import cupy as np
    if np.cuda.is_available():
        _USE_GPU = True
        print("🚀 CuPy GPU detected – using accelerated kernels")
    else:
        raise ImportError("CuPy not available, only use_gpu=False will work")
except (ModuleNotFoundError, ImportError) as e:
    import numpy as np
    print(e)
    print('Failed to import CuPy. Falling back to CPU')
    _USE_GPU = False

syspath = list[str]
imgshape = tuple[int, int]
imge = np.ndarray

executor = cfutures.ThreadPoolExecutor(max_workers=os.cpu_count()//2) # prefer physical cores

class Util:
    @staticmethod
    def readPNGs(paths: list[str]) -> tuple[map, list[tuple[syspath, str, str, imgshape]]] | None:
        def getImageInfo(path: str) -> tuple[syspath, str, str, imgshape]:
            import PIL.Image
            img = PIL.Image.open(path)
            w, h = img.size
            f = img.format
            img.close()
            t = path.split('/')
            t, name = t[:-1], t[-1].split('.')[0]
            return t, name, f, (w, h)
        import skimage.io as ski
        # size verification via metadata
        meta = [getImageInfo(f) for f in paths]  # async later
        if len(set([i[3] for i in meta])) > 1:
            return None
        # png?
        if meta[0][2] != 'PNG':
            return None

        return map(lambda p: ski.imread(p), paths), meta

    @staticmethod
    def _writePNGs(img: imge, path: syspath, name: str) -> None:
        import PIL.Image
        im = img.get()
        im = PIL.Image.fromarray(im)
        im.save(f'{'/'.join(path, )}/_{name}.png', format='PNG', compress_level=9)  # compress as much as possible

    @staticmethod
    async def writePNGs(img: imge, path: syspath, name: str):
        # schedule
        future = executor.submit(Util._writePNGs, img, path, name)
        return future


    # converts 3D to 2D, and back
    @staticmethod
    def flatten(a: imge) -> imge:
        return a.view(dtype=np.uint32).reshape(a.shape[:2])
    @staticmethod
    def fatten(a: imge, epahs) -> imge:
        return a.view(dtype=np.uint8).reshape(epahs)

class Compat: # also routes

    pass

class GPUKernel:
    @staticmethod
    def processImg(match: imge, im: imge, shape: tuple[int,int,int]) -> imge:
        r = np.empty_like(im)
        GPUKernel.transparentMask(match, im, r)
        return Util.fatten(r,shape)

    # custom kernel
    transparentMask = np.ElementwiseKernel(
        'uint32 x, uint32 y',
        'uint32 out',
        '''
        out = x == y ? 0x00000000u & x : y; // ARGB spec, blanks out bytes if equal, keeps y if not (designed to retain transparency for image reconstruction)
        ''',
        'transparentMask'
    )
    @staticmethod
    def batcher(ims: map, i: int, batchSize: int, ln: int, paths: list[str]) -> list[imge]:
        cache: list[imge] = []
        for _ in paths[i: min(i + batchSize, ln)]:
            cache.append(Util.flatten(np.asarray(next(ims))))
        return cache

    @staticmethod
    def batchProcess():
        pass

    pass

def run(paths: list[str], batchSize: int):
    # setup
    ims, meta = Util.readPNGs(paths)
    ln = len(paths)

    subRoots: list[int] = [] # 1st subroot is global root
    i = 0

    while i < ln:
        img = next(ims) # not be processed
        subRoot = np.asarray(img)
        shape = subRoot.shape
        subRoot = Util.flatten(subRoot)
        subRoots.append(i)
        watch = meta[i][1] # debug var
        i += 1
        # gather batch, load into GPU mem
        cache = GPUKernel.batcher(ims, i, batchSize, ln, paths)
        # batch process
        for s in cache:
            r = GPUKernel.processImg(subRoot, s, shape)
            asyncio.run(Util.writePNGs(r, meta[i][0], meta[i][1])) # schedules file write into pool, main thread continues
            i += 1

    cache
    # link subroots together





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    files = [
    ]
    batchSize = 30
    run(files, batchSize-1)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
