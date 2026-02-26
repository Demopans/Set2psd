import asyncio, concurrent.futures as cfutures, numpy as np, os, PIL.Image

# GPU / CPU fallback
try:
    import cupy as xp
    if xp.cuda.is_available():
        print("🚀 CuPy GPU detected – using accelerated kernels")
        _USE_GPU = True
    else:
        raise ImportError("CuPy not available, only use_gpu=False will work")
except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp
    print(e)
    print('Failed to import CuPy. Falling back to CPU')
    _USE_GPU = False

syspath = list[str]
imgshape = tuple[int, int]
image = np.ndarray[tuple[int,int,int],xp.uint8] # explicit cross CPU GPU compat
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
    def _writePNGs(img: image, path: syspath, name: str) -> None:
        import PIL.Image
        im = PIL.Image.fromarray(img)
        im.save(f'{'/'.join(path, )}/_{name}.png', format='PNG', compress_level=9)  # compress as much as possible

    @staticmethod
    async def writePNGs(img: image, path: syspath, name: str):
        # schedule
        future = executor.submit(Util._writePNGs, img, path, name)
        return future


    # converts 3D to 2D, and back
    @staticmethod
    def flatten(a: image) -> image:
        return a.view(dtype=xp.uint32).reshape(a.shape[:2])
    @staticmethod
    def fatten(a: image, epahs) -> image:
        return a.view(dtype=xp.uint8).reshape(epahs)

    @staticmethod
    def batcher(ims: map, i: int, batchSize: int, ln: int, paths: list[str]) -> list[image]:
        cache: list[image] = []
        for _ in paths[i: min(i + batchSize, ln)]:
            cache.append(Util.flatten(xp.asarray(next(ims))))
        return cache

class Compat: # also routes

    pass

class GPUKernel:
    @staticmethod
    def processImg(match: image, im: image, shape: tuple[int,int,int]) -> image:
        r = xp.empty_like(im)
        GPUKernel.transparentMask(match, im, r)
        return Util.fatten(r,shape).get() # loads into system RAM

    # custom kernel
    transparentMask = xp.ElementwiseKernel(
        'uint32 x, uint32 y',
        'uint32 out',
        '''
        out = x == y ? 0x00000000u & x : y; // ARGB spec, blanks out bytes if equal, keeps y if not (designed to retain transparency for image reconstruction)
        ''',
        'transparentMask'
    )

    @staticmethod
    def runner(batchSize: int, ims, meta, paths: list[str]) -> list[int]:
        ln = len(paths)
        subRoots: list[int] = []  # 1st subroot is global root
        i = 0
        while i < ln:
            img = next(ims)  # not be processed
            subRoot = xp.asarray(img)
            shape = subRoot.shape
            subRoot = Util.flatten(subRoot)
            subRoots.append(i)
            i += 1
            # gather batch, load into GPU mem
            cache = Util.batcher(ims, i, batchSize, ln, paths)
            # batch process, make async in the future
            for s in cache:
                r: image = GPUKernel.processImg(subRoot, s, shape)
                asyncio.run(
                    Util.writePNGs(r, meta[i][0], meta[i][1]))  # schedules file write into pool, main thread continues
                i += 1
        return subRoots

    @staticmethod # internal processor
    def batchProcess(paths: list[str], batchSize: int):
        ims, meta = Util.readPNGs(paths)
        subRoots = GPUKernel.runner(batchSize, ims, meta, paths)

        path = list(np.asarray(paths)[subRoots])

        ims, meta = Util.readPNGs(path)
        subRoots = GPUKernel.runner(batchSize, ims, meta, path)

        ims, meta = Util.readPNGs(paths[0:1])
        ins = next(ims)
        asyncio.run(Util.writePNGs(ins, meta[0][0], meta[0][1]))
        path

class CPUKernel:
    @staticmethod
    def processImg(match: image, im: image, shape: tuple[int, int, int]) -> image:
        r = xp.empty_like(im)
        GPUKernel.transparentMask(match, im, r)
        return Util.fatten(r, shape).get()  # loads into system RAM

    @staticmethod
    def runner(batchSize: int, ims, meta, paths: list[str]) -> list[int]:
        ln = len(paths)
        subRoots: list[int] = []  # 1st subroot is global root
        i = 0
        while i < ln:
            img = next(ims)  # not be processed
            subRoot = xp.asarray(img)
            shape = subRoot.shape
            subRoot = Util.flatten(subRoot)
            subRoots.append(i)
            i += 1
            # gather batch, load into GPU mem
            cache = Util.batcher(ims, i, batchSize, ln, paths)
            # batch process, make async in the future
            for s in cache:
                r: image = GPUKernel.processImg(subRoot, s, shape)
                asyncio.run(
                    Util.writePNGs(r, meta[i][0],
                                   meta[i][1]))  # schedules file write into pool, main thread continues
                i += 1
        return subRoots

    @staticmethod  # internal processor
    def batchProcess(paths: list[str], batchSize: int):
        ims, meta = Util.readPNGs(paths)
        subRoots = GPUKernel.runner(batchSize, ims, meta, paths)

        path = list(np.asarray(paths)[subRoots])

        ims, meta = Util.readPNGs(path)
        subRoots = GPUKernel.runner(batchSize, ims, meta, path)

        ims, meta = Util.readPNGs(paths[0:1])
        ins = next(ims)
        asyncio.run(Util.writePNGs(ins, meta[0][0], meta[0][1]))

def run(paths: list[str], batchSize: int):
    pass


