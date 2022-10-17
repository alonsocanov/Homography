import numpy as np
import cv2
import time

import image_processing as image
import utils
from log import Log


my_log = Log('log')
my_log.config()


def main():
    my_log.message('info', '***** Start Homography *****')
    args = utils.getArguments()
    # read image
    img_1 = image.readImg(args.image_1)
    img_2 = image.readImg(args.image_2)
    # gray image
    gray_1 = image.imgGray(img_1)
    gray_2 = image.imgGray(img_2)
    # dimensions
    h_1, w_1 = img_1.shape[:2]
    img_2 = image.resize(img_2, (w_1, h_1))
    # testing
    w, h, f = image.factor((w_1, h_1))
    gray_1 = image.resize(gray_1, (w, h))
    gray_2 = image.resize(gray_2, (w, h))
    # testing

    # descriptor
    descriptor = cv2.BRISK_create()
    # key points and descriptors with BRISK
    kp_1, des1 = descriptor.detectAndCompute(gray_1, None)
    kp_2, des2 = descriptor.detectAndCompute(gray_2, None)
    # index parameters specifies the type of algorithm and number of trees
    # test for homography
    lines = [90, 780, 90 + 1525, 780 + 440]
    cv2.rectangle(img_2, (lines[0], lines[1]),
                  (lines[2], lines[3]), (0, 255, 0), 3)

    start_time = time.time()
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, trees=5)
    # search parameters specifies the number of times that the trees will be recursively traversed
    search_params = dict(checks=50)
    # Fast Library for Approximate Nearest Neighbors
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # knnMatch gives the couples of the best matches
    matches = flann.knnMatch(des1, des2, k=2)
    total_time = time.time() - start_time
    my_log.message('info', ['Match time:', total_time])
    # store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    my_log.message('info', ["Number of good matches: ", len(good)])
    # if number of useful matches
    if len(good) > args.min_match:
        # source points
        dst_pts = np.float32(
            [kp_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / f
        # destination points
        src_pts = np.float32(
            [kp_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) / f
        # homagraphy matrix and matches mask
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        my_log.message('info', 'Homography matrix')
        my_log.message('info', str(H))

    else:
        msg = 'Not enough matches found - {}/{}'.format(
            len(good), args.min_match)
        my_log.message('info', msg)
        matches_mask = None
    if args.show_match:
        # convert mask matches to list
        matches_mask = mask.ravel().tolist()
        # image corners
        pts = np.float32(
            [[0, 0], [0, h_1 - 1], [w_1 - 1, h_1 - 1], [w_1 - 1, 0]]).reshape(-1, 1, 2)
        # distance of current view to draw bounding box
        dst = cv2.perspectiveTransform(pts, H)
        # draw lines for bounding box on image
        # img_2 = cv2.polylines(img_2, [np.int32(dst)],
        #                       True, 255, 3, cv2.LINE_AA)
        img_matches = image.drawMatches(
            gray_1, kp_1, gray_2, kp_2, good, matches_mask)
        # use homagrapphy matrix for to warp image
        img_warp = image.wrapImg(img_2, H, (w_1, h_1))

        cv2.rectangle(img_warp, (lines[0], lines[1]),
                      (lines[2], lines[3]), (0, 255, 0), 3)

        # concatenate image matches and image warp on horizontal axis
        # img_transform = np.concatenate((img_matches, img_warp), axis=1)
        # show image
        image.showImage(img_matches)


if __name__ == "__main__":
    main()
