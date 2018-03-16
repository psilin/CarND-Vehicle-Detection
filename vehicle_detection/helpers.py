
import cv2
import numpy as np
import os
import pickle

from skimage.feature import hog
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def calibrateCamera(path):
    """
    @brief all camera calibration related stuff
    """
    nx = 9
    ny = 6
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

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
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

    return ret, mtx, dist


def warp_matrix():
    """
    @brief get warp matrix
    """
    p1 = (315, 650)
    p2 = (1005, 650)
    p3 = (525, 500)
    p4 = (765, 500)

    src = np.float32([p1, p2, p3, p4])
    dst = np.float32([p1, p2, (p1[0], 360), (p2[0], 360)])

    img_size = (1280, 720)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features


# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_color_features(imgs, cspace='HLS', spatial_size=(8, 8), hist_bins=32, hist_range=(0, 256)):
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
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_hog_features(imgs, cspace='HLS', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
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
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


def prepareData(veh_dir, non_veh_dir, color_space, hog_channel):
    """
    @brief prepare all car and non-car files
    """
    vehicle_files = []
    for dir in os.listdir(veh_dir):
        for file in os.listdir(veh_dir + '/' + dir):
            vehicle_files.append(veh_dir + '/' + dir + '/' + file)

    #print(len(vehicle_files), vehicle_files[100])
    color_features_veh = extract_color_features(vehicle_files, cspace=color_space)
    hog_features_veh = extract_hog_features(vehicle_files, cspace=color_space, hog_channel=hog_channel)
    car_features_veh = [np.concatenate((x, y)) for (x, y) in zip(color_features_veh, hog_features_veh)]
    #print(len(car_features_veh), car_features_veh[100])

    non_vehicle_files = []
    for dir in os.listdir(non_veh_dir):
        for file in os.listdir(non_veh_dir + '/' + dir):
            non_vehicle_files.append(non_veh_dir + '/' + dir + '/' + file)

    #print(len(non_vehicle_files), non_vehicle_files[100])
    color_features_non_veh = extract_color_features(non_vehicle_files, cspace=color_space)
    hog_features_non_veh = extract_hog_features(non_vehicle_files, cspace=color_space, hog_channel=hog_channel)
    car_features_non_veh = [np.concatenate((x, y)) for (x, y) in zip(color_features_non_veh, hog_features_non_veh)]
    #print(len(car_features_non_veh), car_features_non_veh[100])

    # Create an array stack of feature vectors
    X = np.vstack((car_features_veh, car_features_non_veh)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features_veh)), np.zeros(len(car_features_non_veh))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Fit a per-column scaler only on the training data
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    #print(len(X_train), len(X_test), len(y_train), len(y_test))
    #print(X_train[100], y_train[100])
    return X_train, y_train, X_test, y_test


def choose_svm(X_train, y_train, X_test, y_test):
    parameters = {'kernel':('linear', 'rbf'), 'C':[5, 10, 15]}
    svr = SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    print(clf.grid_scores_)
    pred = clf.predict(X_test)
    acc = accuracy_score(pred, y_test)
    print(acc)


def prepare_svm(X_train, y_train, X_test, y_test, kernel='rbf', C=5):
    clf = SVC(kernel=kernel, C=C)
    clf.fit(X_train, y_train)

    filename = 'model.sv'
    pickle.dump(clf, open(filename, 'wb'))

    model = pickle.load(open(filename, 'rb'))
    pred = model.predict(X_test)
    acc = accuracy_score(pred, y_test)
    #pred_train = clf.predict(X_train)
    #acc_train = accuracy_score(pred_train, y_train)
    print(acc)
    #print(acc_train)
    return model


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched  
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate each window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def window_search(binary_warped, low_pass):
    """
    @brief apply window search to find lane on binary warped image
    """

def make_pipeline(M, Minv, mtx, dist, low_pass):
    """
    @brief pipeline closure
    """
    def pipeline(img):
        """
        @brief pipeline takes RGB img returns RGB img with line
        """
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        undistorted_masked = apply_mask(undistorted)
        img_size = (undistorted.shape[1], undistorted.shape[0])
        warped = cv2.warpPerspective(undistorted_masked, M, img_size, flags=cv2.INTER_LINEAR)
        unwarped = cv2.warpPerspective(lane_img, Minv, img_size, flags=cv2.INTER_LINEAR)
        unwarped = cv2.addWeighted(undistorted, 1., unwarped, 0.3, 0)
        return unwarped

    return pipeline

