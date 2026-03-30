from util import Runner
# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    root = 'proc'
    files = [
        '122788001_p0.png',
        '122788001_p1.png'
    ]
    batchSize = 2
    Runner.main(root, files, batchSize, 'Puffo')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
