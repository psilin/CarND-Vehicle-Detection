
from scipy.ndimage.measurements import label
import cv2
import helpers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time


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


def test_features_prep(veh_dir, non_veh_dir):
    """
    @brief test various feature sets with kinear SVM C=10
    """
    for color_space in ('RGB', 'HSV', 'LUV', 'HLS', 'YUV'):
        for hog_channel in (0, 1, 2, 'ALL'):
            for orient in (6, 9, 12):
                t1 = time.time()
                X_train, y_train, X_test, y_test, X_scaler = helpers.prepareData(veh_dir, non_veh_dir, color_space=color_space, hog_channel=hog_channel, orient=orient)
                t2 = time.time()
                clf = helpers.prepare_svm(X_train, y_train, X_test, y_test, X_scaler, kernel='linear', C=10)
                t3 = time.time()
                print(t3 - t2, t2 - t1, hog_channel, color_space, orient)


def test_data_prep(veh_dir, non_veh_dir, color_space):
    """
    @brief test of data preparation functions
    """
    t1 = time.time()
    X_train, y_train, X_test, y_test, X_scaler = helpers.prepareData(veh_dir, non_veh_dir, color_space=color_space, hog_channel='ALL', orient=12)
    t2 = time.time()
    print(t2 - t1)
    clf = helpers.prepare_svm(X_train, y_train, X_test, y_test, X_scaler)
    #clf = helpers.choose_svm(X_train, y_train, X_test, y_test, X_scaler)
    t3 = time.time()
    print(t3 - t2)


def test_find_cars(test_img_dir, cspace): 
    """
    @brief test find cars
    """
    model, X_scaler = pickle.load(open('modelYUV.sv', 'rb'))
    for file in os.listdir(test_img_dir):
        if file[0] == '.': continue

        imgfile = test_img_dir + '/' + file
        print(imgfile)
        image = cv2.imread(imgfile)
        if cspace != 'BGR':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif cspace == 'RGB':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            feature_image = np.copy(image)
        rectangles0 = helpers.find_cars(feature_image, 400, 464, 1., model, X_scaler)
        rectangles1 = []#helpers.find_cars(feature_image, 400, 480, 1.25, model, X_scaler)
        rectangles2 = helpers.find_cars(feature_image, 400, 528, 1.5, model, X_scaler)
        rectangles3 = []#helpers.find_cars(feature_image, 400, 656, 1.75, model, X_scaler)
        rect = rectangles0+rectangles1+rectangles2+rectangles3

        #for rectangle in rect:
        #    cv2.rectangle(image,rectangle[0],rectangle[1],(0,0,255),6) 

        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = helpers.add_heat(heat, rect)
        heat = helpers.apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        image = helpers.draw_labeled_bboxes(np.copy(image), labels)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title('windows')
        plt.show()


def test_make_pipeline(path, M, Minv, mtx, dist):
    """
    @brief test of make pipeline function
    """

