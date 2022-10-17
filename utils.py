import argparse
import glob
import os


def getArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-1", type=str, default='data/Sura_Poliza_Blank_1.jpg',
                        help='first file path to to analyze it must be a .jpg file')
    parser.add_argument("--image-2", type=str, default='data/Sura_Poliza_Auto_75_2.jpg',
                        help='second file path to to analyze it must be a .jpg file')
    parser.add_argument('--min-match', type=int, default=50,
                        help='set minimum match count')
    parser.add_argument('--show-img', type=int, default=0,
                        help='set to 1 to show images')
    parser.add_argument('--show-match', type=int, default=0,
                        help='set to 1 to show images')
    parser.add_argument('--wait', type=int, default=0,
                        help='set integer number of mili-seconds between images showing')
    return parser.parse_args()


def getFilesinDir(path: str) -> list:
    return glob.glob(path, recursive=True)


def createDirNoLog(dir_path: str) -> bool:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return True


def checkPath(dir_path: str) -> bool:
    if not os.path.exists(dir_path):
        return False
    return True


def createPath(*file_path):
    # join full path
    return os.path.join(*file_path)


def getFileDirName(file_path: str) -> tuple:
    dir_path, file_name_ext = os.path.split(file_path)
    return dir_path, file_name_ext


def getFileNameExt(file_path: str) -> tuple:
    _, file_name_ext = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(file_name_ext)
    return file_name, file_ext
