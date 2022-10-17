import numpy as np
import cv2


def showImage(img: np.ndarray, img_name: str = 'Image', wait: int = 0, loc_x: int = 20, loc_y: int = 20) -> None:
    h, w = img.shape[:2]
    w, h, _ = factor((w, h))
    img = resize(img, (w, h))
    cv2.namedWindow(img_name)
    cv2.moveWindow(img_name, loc_x, loc_y)
    cv2.imshow(img_name, img)
    cv2.waitKey(wait)
    cv2.destroyAllWindows()


def readImg(path: str) -> np.ndarray:
    return cv2.imread(path)


def imgGray(img: np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def factor(dim: tuple, factor: float = 1.0):
    max_w, max_h = 1000, 1000
    w, h = dim
    if factor == 1.0:
        if h > max_h:
            factor = max_h / h
        if w * factor > max_w:
            factor = max_w / w
    return int(w * factor), int(h * factor), factor


def resize(img: np.ndarray, dim: tuple) -> np.ndarray:
    w, h = dim
    return cv2.resize(img, (w, h))


def drawMatches(img_1, kp_1, img_2, kp_2, matches, matches_mask) -> np.ndarray:
    # parameters to draw matches
    draw_params = dict(matchColor=(
        0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
    # resulting image with visible matches
    img_matches = cv2.drawMatches(
        img_1, kp_1, img_2, kp_2, matches, None, **draw_params)
    return img_matches


def wrapImg(img, matrix, dim) -> np.ndarray:
    return cv2.warpPerspective(img, matrix, dim)
