
import cv2
import helpers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

def test_camera_calibration(path, mtx, dist):
    """
    @brief test camera calibration
    """
    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            undistorted = cv2.undistort(img, mtx, dist, None, mtx)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(undistorted)
            ax2.set_title('Undistorted Image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()


def test_warping(path, M, mtx, dist):
    """
    @brief test images warping given test images dir
    """
    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            undistorted = cv2.undistort(img, mtx, dist, None, mtx)
            undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            p1 = (315, 650)
            p2 = (1005, 650)
            p3 = (525, 500)
            p4 = (765, 500)
            cv2.line(undistorted, p1, p2, (255, 0, 0), 5)
            cv2.line(undistorted, p3, p4, (255, 0, 0), 5)
            cv2.line(undistorted, p1, p3, (255, 0, 0), 5)
            cv2.line(undistorted, p2, p4, (255, 0, 0), 5)

            pr1 = (100, 720)
            pr2 = (1200, 720)
            pr3 = (550, 450)
            pr4 = (750, 450)
            cv2.line(undistorted, pr1, pr2, (0, 255, 255), 5)
            cv2.line(undistorted, pr3, pr4, (0, 255, 255), 5)
            cv2.line(undistorted, pr1, pr3, (0, 255, 255), 5)
            cv2.line(undistorted, pr2, pr4, (0, 255, 255), 5)

            undistorted = helpers.apply_mask(undistorted)

            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            img_size = (gray.shape[1], gray.shape[0])
            warped = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(undistorted, cmap='gray')
            ax1.set_title('Original Undistorted Image', fontsize=50)
            ax2.imshow(warped, cmap='gray')
            ax2.set_title('Warped Image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()


def test_make_pipeline(path, M, Minv, mtx, dist):
    """
    @brief test of make pipeline function
    """
    pipeline = helpers.make_pipeline(M, Minv, mtx, dist, None)

    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res_img = pipeline(img)

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(res_img)
            ax2.set_title('Result Image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()

