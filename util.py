import concurrent.futures as cfutures, os, cv2, PIL.Image as PILI, numpy as np, py7zr

from kernel import CPUKernel, GPUKernel

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
    import numpy as xp

    print(e)
    print('Failed to import CuPy. Falling back to CPU')
    _USE_GPU = False

bite = { 1: xp.uint8, 2: xp.uint16, 4: xp.uint32, 8: xp.uint64 }
cDepth, bitDepth = bite[2], bite[8]
syspath = list[str]
imgshape = tuple[int, int]
image = xp.ndarray[tuple[int, int, int], cDepth]  # explicit cross CPU GPU compat
executor = cfutures.ThreadPoolExecutor(max_workers=os.cpu_count() // 2)  # prefer physical cores


class Util:
    class IO: # contains generally blocking calls
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


    # converts 3D to 2D, and back
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

        @staticmethod
        def wipe(root: str, name: str, method: str):
            py7zr.SevenZipFile(f"{root}/{name}.7z", 'w', filters=[{"id": Util.Compress.method[method], "preset": 7}])
        @staticmethod
        def compress(root: str, files: list[str], name: str, method: str):
            with py7zr.SevenZipFile(f"{root}/{name}.7z", 'a', filters=[{"id": Util.Compress.method[method] , "preset": 7}]) as archive:
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
    def main(root: str, files: list[str], batchSize: int, compress: str = 'lzma'):
        import operator
        # verify
        if len(meta := set([Util.IO.getImageInfo(f'{root}/{_}') for _ in files])) != 1:
            exit("inconsistent size/bit depth")
        elif (meta := list(meta)[0])[0] != 'PNG':
            exit("not PNG")

        # generate tree
        tree = Util.makeTree(files, batchSize)
        batches = [
            tree[i * batchSize:min(len(files), i * batchSize + batchSize)] for i in range(len(files) // batchSize)
        ]
        # feed GPU loop by batch
        for b in batches:
            f = files[b[0][-1]:b[-1][-1]+1]
            res = Runner.runner(root, f)
            # schedule write before next loop
            [Util.IO.writePNG(root,file,_) for file,_ in zip(f[1:],res)]

        f = operator.itemgetter(*[_[0][-1] for _ in batches])(files)
        res = Runner.runner(root, f)
        batches, b = [Util.IO.writePNG(root, file, _) for file, _ in zip(f[1:], res)], Util.IO.writePNG(root,files[0],Util.IO.readPNG(root,files[0]).result())
        # write metadata
        Util.IO.writeMeta(root, files, tree).result()
        [_.result() for _ in batches]
        b.result()
        # pack
        Util.Compress.wipe(root, 'Puffo', compress)
        Util.Compress.compress(root, files, 'Puffo', compress)
        # rm leftover files


if __name__ == '__main__':
    root = 'proc/4 PuffoTim/Puffo'
    files = [
        'Puffo_0001.png',
        'Puffo_0002.png',
        'Puffo_0003.png',
        'Puffo_0004.png',
        'Puffo_0005.png',
        'Puffo_0006.png',
        'Puffo_0007.png',
        'Puffo_0008.png',
        'Puffo_0009.png',
        'Puffo_0010.png',
        'Puffo_0011.png',
        'Puffo_0012.png',
        'Puffo_0013.png',
        'Puffo_0014.png',
        'Puffo_0015.png',
        'Puffo_0016.png',
        'Puffo_0017.png',
        'Puffo_0018.png',
        'Puffo_0019.png',
        'Puffo_0020.png',
        'Puffo_0021.png',
        'Puffo_0022.png',
        'Puffo_0023.png',
        'Puffo_0024.png',
        'Puffo_0025.png',
        'Puffo_0026.png',
        'Puffo_0027.png',
        'Puffo_0028.png',
        'Puffo_0029.png',
        'Puffo_0030.png',
        'Puffo_0031.png',
        'Puffo_0032.png',
        'Puffo_0033.png',
        'Puffo_0034.png',
        'Puffo_0035.png',
        'Puffo_0036.png',
        'Puffo_0037.png',
        'Puffo_0038.png',
        'Puffo_0039.png',
        'Puffo_0040.png',
        'Puffo_0041.png',
        'Puffo_0042.png',
        'Puffo_0043.png',
        'Puffo_0044.png',
        'Puffo_0045.png',
        'Puffo_0046.png',
        'Puffo_0047.png',
        'Puffo_0048.png',
        'Puffo_0049.png',
        'Puffo_0050.png',
        'Puffo_0051.png',
        'Puffo_0052.png',
        'Puffo_0053.png',
        'Puffo_0054.png',
        'Puffo_0055.png',
        'Puffo_0056.png',
        'Puffo_0057.png',
        'Puffo_0058.png',
        'Puffo_0059.png',
        'Puffo_0060.png',
        'Puffo_0061.png',
        'Puffo_0062.png',
        'Puffo_0063.png',
        'Puffo_0064.png',
        'Puffo_0065.png',
        'Puffo_0066.png',
        'Puffo_0067.png',
        'Puffo_0068.png',
        'Puffo_0069.png',
        'Puffo_0070.png',
        'Puffo_0071.png',
        'Puffo_0072.png',
        'Puffo_0073.png',
        'Puffo_0074.png',
        'Puffo_0075.png',
        'Puffo_0076.png',
        'Puffo_0077.png',
        'Puffo_0078.png',
        'Puffo_0079.png',
        'Puffo_0080.png',
        'Puffo_0081.png',
        'Puffo_0082.png',
        'Puffo_0083.png',
        'Puffo_0084.png',
        'Puffo_0085.png',
        'Puffo_0086.png',
        'Puffo_0087.png',
        'Puffo_0088.png',
        'Puffo_0089.png',
        'Puffo_0090.png',
        'Puffo_0091.png',
        'Puffo_0092.png',
        'Puffo_0093.png',
        'Puffo_0094.png',
        'Puffo_0095.png',
        'Puffo_0096.png',
        'Puffo_0097.png',
        'Puffo_0098.png',
        'Puffo_0099.png',
        'Puffo_0100.png',
        'Puffo_0101.png',
        'Puffo_0102.png',
        'Puffo_0103.png',
        'Puffo_0104.png',
        'Puffo_0105.png',
        'Puffo_0106.png',
        'Puffo_0107.png',
        'Puffo_0108.png',
        'Puffo_0109.png',
        'Puffo_0110.png',
        'Puffo_0111.png',
        'Puffo_0112.png',
        'Puffo_0113.png',
        'Puffo_0114.png',
        'Puffo_0115.png',
        'Puffo_0116.png',
        'Puffo_0117.png',
        'Puffo_0118.png',
        'Puffo_0119.png',
        'Puffo_0120.png'
    ]
    batchSize = 30
    Runner.main(root, files, batchSize)
