
from scipy.ndimage.measurements import label
import cv2
import numpy as np
import os
import pickle
# import time

from skimage.feature import hog
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class RingBuffer:
    """
    @brief has some memory of found recatngles through length last frames
    """
    def __init__(self, length):
        self.length = length
        self.rectangles_list = []

    def add(self, rectangles):
        if len(self.rectangles_list) > self.length:
            self.rectangles_list.pop(0)

        self.rectangles_list.append(rectangles)

    def get(self):
        return [item for sublist in self.rectangles_list for item in sublist]


def calibrateCamera(path):
    """
    @brief all camera calibration related stuff
    """
    nx = 9
    ny = 6
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, : 2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    shape = None
    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            shape = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       shape, None, None)

    return ret, mtx, dist


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0],
                                    channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_color_features(imgs, cspace, spatial_size=(8, 8), hist_bins=32,
                           hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
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
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins,
                                   bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell,
                     cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis is True:
        (features,
         hog_image) = hog(img, orientations=orient,
                          pixels_per_cell=(pix_per_cell, pix_per_cell),
                          cells_per_block=(cell_per_block, cell_per_block),
                          block_norm='L2-Hys', transform_sqrt=True,
                          visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys', transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_hog_features(imgs, cspace, orient=12,
                         pix_per_cell=16, cell_per_block=2, hog_channel='ALL'):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'BGR':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            elif cspace == 'RGB':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(
                                    feature_image[:, :, channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(
                        feature_image[:, :, hog_channel], orient,
                        pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


def prepareData(veh_dir, non_veh_dir, color_space, hog_channel, orient):
    """
    @brief prepare all car and non-car files
    """
    vehicle_files = []
    for dir in os.listdir(veh_dir):
        for file in os.listdir(veh_dir + '/' + dir):
            vehicle_files.append(veh_dir + '/' + dir + '/' + file)

    # print(len(vehicle_files), vehicle_files[100])
    # color_features_veh = extract_color_features(vehicle_files,
    #                                             cspace=color_space)
    hog_features_veh = extract_hog_features(vehicle_files,
                                            cspace=color_space,
                                            hog_channel=hog_channel,
                                            orient=orient)
    # car_features_veh = [np.concatenate(
    #        (x, y)) for (x, y) in zip(color_features_veh, hog_features_veh)]
    # print(len(car_features_veh), car_features_veh[100])

    non_vehicle_files = []
    for dir in os.listdir(non_veh_dir):
        for file in os.listdir(non_veh_dir + '/' + dir):
            non_vehicle_files.append(non_veh_dir + '/' + dir + '/' + file)

    # print(len(non_vehicle_files), non_vehicle_files[100])
    # color_features_non_veh = extract_color_features(non_vehicle_files,
    #                                                  cspace=color_space)
    hog_features_non_veh = extract_hog_features(non_vehicle_files,
                                                cspace=color_space,
                                                hog_channel=hog_channel,
                                                orient=orient)
    # car_features_non_veh = [np.concatenate(
    #    (x, y)) for (x, y) in zip(color_features_non_veh,
    #                              hog_features_non_veh)]
    # print(len(car_features_non_veh), car_features_non_veh[100])

    # Create an array stack of feature vectors
    X = np.vstack((hog_features_veh, hog_features_non_veh)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(hog_features_veh)),
                   np.zeros(len(hog_features_non_veh))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    (X_train, X_test, y_train,
     y_test) = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    # (X_train, X_test, y_train,
    #  y_test) = train_test_split(X, y, test_size=0.2, random_state=1)

    # Fit a per-column scaler only on the training data
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    # print(len(X_train), len(X_test), len(y_train), len(y_test))
    # print(X_train[100], y_train[100])
    return X_train, y_train, X_test, y_test, X_scaler


def choose_svm(X_train, y_train, X_test, y_test, X_scaler):
    parameters = {'kernel': ('linear', 'rbf'), 'C': [5, 10, 15]}
    svr = SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    print(clf.grid_scores_)
    pred = clf.predict(X_test)
    acc = accuracy_score(pred, y_test)
    print(acc)


def prepare_svm(X_train, y_train, X_test, y_test,
                X_scaler, kernel='linear', C=10):
    clf = SVC(kernel=kernel, C=C)
    clf.fit(X_train, y_train)

    filename = 'model.sv'
    pickle.dump((clf, X_scaler), open(filename, 'wb'))

    model, X_scaler = pickle.load(open(filename, 'rb'))
    pred = model.predict(X_test)
    acc = accuracy_score(pred, y_test)
    print(acc)
    return model, X_scaler


# Define a single function that can extract features using hog sub-sampling
# and make predictions (img=cv2.imread(...))
def find_cars(img, ystart, ystop, scale, svc, X_scaler,
              orient=12, pix_per_cell=16, cell_per_block=2,
              spatial_size=(8, 8), hist_bins=32):
    """
    @brief fing cars with sliding window, ystart = 400, ystop = 656
    """

    ctrans_tosearch = img[ystart:ystop, :, :]
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1]/scale),
                                      np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    # nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)

    rectangles = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            # t1 = time.time()
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # t2 = time.time()
            # Extract the image patch
            # subimg = cv2.resize(
            #    ctrans_tosearch[ytop:ytop+window,
            #                    xleft:xleft+window], (64, 64))

            # Get color features
            # spatial_features = bin_spatial(subimg, size=spatial_size)
            # hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            # test_features = np.hstack(
            #    (spatial_features,
            #        hist_features, hog_features)).reshape(1, -1)
            # t3 = time.time()

            # test_features = X_scaler.transform(np.hstack(
            #        (spatial_features,
            #         hist_features,
            #         hog_features)).reshape(1, -1))
            # t4 = time.time()
            # test_features = X_scaler.transform(np.hstack(
            #         (shape_feat, hist_feat)).reshape(1, -1))
            # t5 = time.time()
            test_features = X_scaler.transform(hog_features.reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left, ytop_draw + ystart),
                                   (xbox_left + win_draw,
                                    ytop_draw + win_draw + ystart)))
                # cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart),
                #                        (xbox_left+win_draw,
                #                            ytop_draw+win_draw+ystart),
                #                        (0, 0, 255), 6)
            # t6 = time.time()
            # print(xb, yb, t6-t5,t5-t4,t4-t3,t3-t2,t2-t1)

    return rectangles


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def make_pipeline(mtx, dist, cspace):
    """
    @brief pipeline closure
    """
    model, X_scaler = pickle.load(open('model.sv', 'rb'))

    ring = RingBuffer(8)

    def pipeline(img):
        """
        @brief pipeline: undistrot img -> conver color space ->
        window search -> heat map with threshold -> draw resulting boxes
        """
        # undistorted = img
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(undistorted, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(undistorted, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(undistorted, cv2.COLOR_RGB2YCrCb)
            elif cspace == 'BGR':
                feature_image = cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR)
        else:
            feature_image = np.copy(undistorted)

        rectangles_list = []
        rectangles_list.append(
            find_cars(feature_image, 400, 480, 1., model, X_scaler))
        rectangles_list.append(
            find_cars(feature_image, 400, 480, 1.25, model, X_scaler))
        rectangles_list.append(
            find_cars(feature_image, 400, 496, 1.5, model, X_scaler))
        rectangles_list.append(
            find_cars(feature_image, 432, 528, 1.5, model, X_scaler))
        rectangles_list.append(
            find_cars(feature_image, 432, 592, 2, model, X_scaler))
        rectangles_list.append(
            find_cars(feature_image, 432, 624, 3, model, X_scaler))
        rectangles_list.append(
            find_cars(feature_image, 400, 656, 4, model, X_scaler))

        flat = [rect for rectangles in rectangles_list for rect in rectangles]

        if len(flat) > 0:
            ring.add(flat)

        heat = np.zeros_like(undistorted[:, :, 0]).astype(np.float)
        flat = ring.get()
        heat = add_heat(heat, flat)
        heat = apply_threshold(heat, 17)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(undistorted, labels)

        # for rect in rectangles0+rectangles1+rectangles2+rectangles3:
        #    cv2.rectangle(img, rect[0], rect[1], (0, 0, 255), 6)

        return draw_img

    return pipeline
