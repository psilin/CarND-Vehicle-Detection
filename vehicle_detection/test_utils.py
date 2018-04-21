
from scipy.ndimage.measurements import label
import cv2
import helpers
# import matplotlib
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


def test_features_prep(veh_dir, non_veh_dir):
    """
    @brief test various feature sets with kinear SVM C=10
    """
    for color_space in ('RGB', 'HSV', 'LUV', 'HLS', 'YUV'):
        for hog_channel in (0, 1, 2, 'ALL'):
            for orient in (6, 9, 12):
                t1 = time.time()
                (X_train, y_train, X_test, y_test,
                    X_scaler) = helpers.prepareData(veh_dir,
                                                    non_veh_dir,
                                                    color_space=color_space,
                                                    hog_channel=hog_channel,
                                                    orient=orient)
                t2 = time.time()
                helpers.prepare_svm(X_train, y_train, X_test, y_test, X_scaler,
                                    kernel='linear', C=10)
                t3 = time.time()
                print(t3 - t2, t2 - t1, hog_channel, color_space, orient)


def test_data_prep(veh_dir, non_veh_dir, color_space):
    """
    @brief test of data preparation functions (trains SVM classifier)
    """
    t1 = time.time()
    (X_train, y_train, X_test, y_test,
        X_scaler) = helpers.prepareData(veh_dir,
                                        non_veh_dir,
                                        color_space=color_space,
                                        hog_channel='ALL',
                                        orient=12)
    t2 = time.time()
    print(t2 - t1)
    helpers.prepare_svm(X_train, y_train, X_test, y_test, X_scaler)
    # helpers.choose_svm(X_train, y_train, X_test, y_test, X_scaler)
    t3 = time.time()
    print(t3 - t2)


def test_find_cars(test_img_dir, cspace):
    """
    @brief test find cars
    """
    model, X_scaler = pickle.load(open('model.sv', 'rb'))
    for file in os.listdir(test_img_dir):
        if file[0] == '.':
            continue

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

        rectangle_list = []
        rectangle_list.append(
            helpers.find_cars(feature_image, 400, 480, 1., model, X_scaler))
        rectangle_list.append(
            helpers.find_cars(feature_image, 400, 480, 1.25, model, X_scaler))
        rectangle_list.append(
            helpers.find_cars(feature_image, 400, 496, 1.5, model, X_scaler))
        rectangle_list.append(
            helpers.find_cars(feature_image, 432, 528, 1.5, model, X_scaler))
        rectangle_list.append(
            helpers.find_cars(feature_image, 432, 592, 2, model, X_scaler))
        rectangle_list.append(
            helpers.find_cars(feature_image, 432, 624, 3, model, X_scaler))
        rectangle_list.append(
            helpers.find_cars(feature_image, 400, 656, 4, model, X_scaler))

        rect = [rects for rectangles in rectangle_list for rects in rectangles]

        # for rectangle in rect:
        #    cv2.rectangle(image,rectangle[0],rectangle[1],(0,0,255),6)

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = helpers.add_heat(heat, rect)
        heat = helpers.apply_threshold(heat, 2)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        image = helpers.draw_labeled_bboxes(np.copy(image), labels)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title('windows')
        plt.show()


def visualize_hog():
    '''
    @brief visualizes HOG features
    '''
    files = ['./../examples/car_image1.png',
             './../examples/car_image2.png',
             './../examples/car_image3.png',
             './../examples/non_car_image1.png',
             './../examples/non_car_image2.png',
             './../examples/non_car_image3.png']
    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        ch1 = image[:, :, 0]
        ch2 = image[:, :, 1]
        ch3 = image[:, :, 2]

        hog1, im1 = helpers.get_hog_features(ch1,
                                             orient=12,
                                             pix_per_cell=16,
                                             cell_per_block=2,
                                             vis=True,
                                             feature_vec=False)
        hog2, im2 = helpers.get_hog_features(ch2,
                                             orient=12,
                                             pix_per_cell=16,
                                             cell_per_block=2,
                                             vis=True,
                                             feature_vec=False)
        hog3, im3 = helpers.get_hog_features(ch3,
                                             orient=12,
                                             pix_per_cell=16,
                                             cell_per_block=2,
                                             vis=True,
                                             feature_vec=False)

        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,
                                                             figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(ch1, cmap='gray')
        ax1.set_title('Original image channel H', fontsize=10)
        ax2.imshow(ch2, cmap='gray')
        ax2.set_title('Original image channel S', fontsize=10)
        ax3.imshow(ch3, cmap='gray')
        ax3.set_title('Original image channel V', fontsize=10)
        ax4.imshow(im1, cmap='gray')
        ax4.set_title('HOG features channel H', fontsize=10)
        ax5.imshow(im2, cmap='gray')
        ax5.set_title('HOG features channel S', fontsize=10)
        ax6.imshow(im3, cmap='gray')
        ax6.set_title('HOG features channel V', fontsize=10)

        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)
        plt.show()


def visualize_windows():
    '''
    @brief shows all possible windows
    '''

    model, X_scaler = pickle.load(open('model.sv', 'rb'))
    files = ['./../test_images/test1.jpg']

    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rectangle_list = []
        # rectangle_list.append(
        #    helpers.find_cars(image, 400, 464, 1., model, X_scaler))
        # rectangle_list.append(
        #    helpers.find_cars(image, 416, 480, 1., model, X_scaler))
        # rectangle_list.append(
        #    helpers.find_cars(image, 400, 480, 1.25, model, X_scaler))
        # rectangle_list.append(
        #    helpers.find_cars(image, 400, 496, 1.5, model, X_scaler))
        # rectangle_list.append(
        #    helpers.find_cars(image, 432, 528, 1.5, model, X_scaler))
        # rectangle_list.append(
        #    helpers.find_cars(image, 432, 560, 2, model, X_scaler))
        # rectangle_list.append(
        #    helpers.find_cars(image, 464, 592, 2, model, X_scaler))
        # rectangle_list.append(
        #    helpers.find_cars(image, 432, 624, 3, model, X_scaler))
        rectangle_list.append(
            helpers.find_cars(image, 400, 656, 4, model, X_scaler))

        rect = [rects for rectangles in rectangle_list for rects in rectangles]

        for rectangle in rect:
            cv2.rectangle(image, rectangle[0], rectangle[1], (0, 0, 255), 6)

        plt.imshow(image)
        plt.title('windows')
        plt.show()
