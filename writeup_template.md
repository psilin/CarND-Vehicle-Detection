## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_image1.png
[image2]: ./output_images/car_image2.png
[image3]: ./output_images/car_image3.png
[image4]: ./output_images/non_car_image1.png
[image5]: ./output_images/non_car_image2.png
[image6]: ./output_images/non_car_image3.png
[image7]: ./output_images/car_image1_hog.png
[image8]: ./output_images/car_image2_hog.png
[image9]: ./output_images/car_image3_hog.png
[image10]: ./output_images/non_car_image1_hog.png
[image11]: ./output_images/non_car_image2_hog.png
[image12]: ./output_images/non_car_image3_hog.png
[image13]: ./output_images/windows_all_1.png
[image14]: ./output_images/windows_all_2.png
[image15]: ./output_images/windows_all_3.png
[image16]: ./output_images/windows_all_4.png
[image17]: ./output_images/windows_all_5.png
[image18]: ./output_images/windows_all_6.png
[image19]: ./output_images/windows_all_7.png
[image20]: ./output_images/windows_all_8.png
[image21]: ./output_images/windows_all_9.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 0. General discussion

Code is provided with writeup.md or available [here](https://github.com/psilin/CarND-Vehicle-Detection/). To run it I used virtual invironment based on `python==3.5.2` with packages needed to be installed are in `requirements.txt` file. Project can be 
installed using following command:
```python
python setup.py install
```

Then `run_vehicle_detection` script should be executed from `vehicle_detection` directory as all paths to pictures and videos are relative to this directory. On the other hand, I just run following command from `lane_line` directory:
```python
python core.py
```
as it is a main entry point to the project. It produces `output_project_video.mp4` from `project_video.mp4` drawing boxes around cars on it.

Project code structure is as follows. `vehicle_detection/core.py` contains handling video and main pipeline execution per frame. `vehicle_detection/helpers.py` contains all methods that are used in pipeline and pipeline function itself. `lane_lines/test_utils.py` contains 
various test, almost each functions from `helpers.py` is tested there. All pictures (except training sets) in this reported were produced by these tests.

### Camera Calibration

It seems that it is not really importtant in this project but I decided to use camera calibration procedure from previous project (CarND-Advanced-Lane-Lines) as I had to work with the same video so it clearly needed that procedure. Though it is not so 
important as I did not need to deal with line cuvatures it this project. Here is how it has been implemented. As has been already mentioned camera calibaration function contains in `helpers.py` file and is called `calibrateCamera()`. I took a standard approach from lesson. Converted image to grayscale. Then called `cv2.findChessboardCorners()` on it and obtained 
image points. Object points are well known as it is a plain `9x6` grid. Computed those 2 subsets for all images in `camera_cal` directory then concatenated all image points and all object points. 2 resulting sets were used in `cv2.calibrateCamera()` function
to obtain parameters needed for camera calibration.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First of all I downloaded vehicles and non-vehicles data and stored it `vehicles` and `non-vehicles` directories. 3 typical examles of vehicle and non-vehicle are shown below.

![alt text][image1] ![alt text][image2] ![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] 

Data preparation for SVM training was done in `prepareData()` function in `helpers.py` file. I used standard approach from lesson and reused `extract_hog_features()` and `extract_color_features()` functions provided in lesson. Later I abandoned `extract_color_features()`
as it gave me too much false positives through video. I must mention it now but the most difficult part about this project was that I had to re-evaluate my whole approach from top to the bottom several times as some choices that looked best by bare numbers would result in
much worse performance later. That is why in the end I decided against using color features although I had better training results using it. So I ended up using HOG features with following parameters:

```python
color_space=`HSV`
hog_channel=`ALL`
orientation=12
pix_per_cell=16
cell_per_block=2
```

Below is 6 images visualizing HOG features I used (each image below is corresponding to image above - 3 cars and 3 non-cars). These images were produced by `visualize_hog()` function from `test_util.py` file.

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]

#### 2. Explain how you settled on your final choice of HOG parameters.

First of all I tried to use HOG features with following parameters (I obatained the best results with it during lesson):

```python
color_space=`HLS`
hog_channel=`ALL`
orientation=9
pix_per_cell=8
cell_per_block=2
```

That coice of parameters gave me reasonably good results but I discovered that my pipeline was too slow, it took 2.5 hours to produce final video using it so I had to fing ways of speeding it up. One of solutions I found was to set `pix_per_cell=16`. I will
discuss it in more detail in later sections. So I decided to stick with `pix_per_cell=16` and `cell_per_block=2` and tried to play with other parameters. I decided to train an SVM classifier with linear kernal and `C=10` and check its performance on test set.
Below is a table of results I got.

| Performance   | Training time(s) | Featurre extraction time (s) | Channel | Color space | Orientations|
|:-------------:|:----------------:|:----------------------------:|:-------:|:-----------:|:-----------:|
|0.974          | 24.6             | 29.1                         | 0       |  RGB        | 6           |
|0.979          | 18.0             | 30.3                         | 0       |  RGB        | 9           |
|0.984          | 15.8             | 31.0                         | 0       |  RGB        | 12          |
|0.978          | 22.1             | 28.9                         | 1       |  RGB        | 6           |
|0.982          | 17.6             | 30.1                         | 1       |  RGB        | 9           |
|0.981          | 15.5             | 31.1                         | 1       |  RGB        | 12          |
|0.983          | 25.6             | 29.0                         | 2       |  RGB        | 6           |
|0.981          | 19.2             | 30.0                         | 2       |  RGB        | 9           |
|0.983          | 15.2             | 30.8                         | 2       |  RGB        | 12          |
|0.981          | 22.9             | 62.9                         | ALL     |  RGB        | 6           |
|0.985          | 25.9             | 65.5                         | ALL     |  RGB        | 9           |
|0.986          | 30.2             | 68.3                         | ALL     |  RGB        | 12          |
|0.970          | 79.4             | 30.8                         | 0       |  HSV        | 6           |
|0.969          | 54.3             | 31.7                         | 0       |  HSV        | 9           |
|0.964          | 34.4             | 32.9                         | 0       |  HSV        | 12          |
|0.977          | 29.2             | 30.6                         | 1       |  HSV        | 6           |
|0.974          | 22.0             | 31.4                         | 1       |  HSV        | 9           |
|0.974          | 17.8             | 32.6                         | 1       |  HSV        | 12          |
|0.986          |  8.9             | 30.2                         | 2       |  HSV        | 6           |
|0.986          |  9.3             | 31.0                         | 2       |  HSV        | 9           |
|0.988          |  9.6             | 31.8                         | 2       |  HSV        | 12          |
|0.988          | 13.1             | 64.9                         | ALL     |  HSV        | 6           |
|0.990          | 17.7             | 67.6                         | ALL     |  HSV        | 9           |
|0.993          | 20.5             | 70.3                         | ALL     |  HSV        | 12          |
|0.984          | 12.2             | 34.2                         | 0       |  LUV        | 6           |
|0.985          | 11.7             | 35.2                         | 0       |  LUV        | 9           |
|0.985          | 11.5             | 36.1                         | 0       |  LUV        | 12          |
|0.976          | 33.6             | 34.5                         | 1       |  LUV        | 6           |
|0.975          | 17.0             | 35.4                         | 1       |  LUV        | 9           |
|0.977          | 17.8             | 36.2                         | 1       |  LUV        | 12          |
|0.977          | 71.7             | 34.6                         | 2       |  LUV        | 6           |
|0.971          | 87.4             | 35.6                         | 2       |  LUV        | 9           |
|0.965          | 43.8             | 36.6                         | 2       |  LUV        | 12          |
|0.990          | 13.0             | 68.8                         | ALL     |  LUV        | 6           |
|0.989          | 17.6             | 71.3                         | ALL     |  LUV        | 9           |
|0.993          | 21.1             | 74.0                         | ALL     |  LUV        | 12          |
|0.971          |124.3             | 31.0                         | 0       |  HLS        | 6           |
|0.966          | 95.0             | 31.8                         | 0       |  HLS        | 9           |
|0.964          | 58.2             | 32.8                         | 0       |  HLS        | 12          |
|0.982          | 10.1             | 30.4                         | 1       |  HLS        | 6           |
|0.984          | 10.5             | 31.1                         | 1       |  HLS        | 9           |
|0.988          | 11.0             | 32.0                         | 1       |  HLS        | 12          |
|0.975          | 70.7             | 30.7                         | 2       |  HLS        | 6           |
|0.971          | 42.4             | 31.6                         | 2       |  HLS        | 9           |
|0.974          | 29.0             | 32.5                         | 2       |  HLS        | 12          |
|0.987          | 14.9             | 64.8                         | ALL     |  HLS        | 6           |
|0.989          | 19.5             | 68.0                         | ALL     |  HLS        | 9           |
|0.992          | 24.1             | 70.6                         | ALL     |  HLS        | 12          |
|0.984          | 10.8             | 29.6                         | 0       |  YUV        | 6           |
|0.984          | 11.6             | 30.7                         | 0       |  YUV        | 9           |
|0.987          | 11.1             | 31.6                         | 0       |  YUV        | 12          |
|0.977          | 37.2             | 29.9                         | 1       |  YUV        | 6           |
|0.975          | 20.8             | 30.8                         | 1       |  YUV        | 9           |
|0.976          | 19.9             | 31.8                         | 1       |  YUV        | 12          |
|0.977          | 99.2             | 30.1                         | 2       |  YUV        | 6           |
|0.973          |154.0             | 31.2                         | 2       |  YUV        | 9           |
|0.966          | 79.6             | 32.1                         | 2       |  YUV        | 12          |
|0.989          | 13.7             | 64.6                         | ALL     |  YUV        | 6           |
|0.990          | 18.7             | 66.9                         | ALL     |  YUV        | 9           |
|0.991          | 22.5             | 69.5                         | ALL     |  YUV        | 12          |

Using this table I decided to try `HLS`, `HSV`, `YUV` and `LUV` color spaces using all channels and orientations set to `12` as it provided the best results. trining time was not important
as it was not directly affecting pipeline and feature extraction time was much faster than classifier decision so it did not matter much (in terms of pipeline execution time). These results 
were produced by `test_features_prep()` function from `test_utils.py` file.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First of all I decided to use SVM as a classifier as it was recomended so during lesson and various Slack discussions. Then I tried using `GridSearchCV()` function from `sklearn.grid_search` (code in `choose_svm()` function in `helpers.py`).
I was choosing between `rbf` and `linear` kernels and various values for `C`. The best results I got with `rbf` kernel and `C=5` but during pipeline profiling I realized that `rbf` kernel was 2 times
slower than `linear` kernal and as classifier decision was by far the slowest single operation in pipeline I was forced to move back to `linear` kernel. So my final parameters were `kernel=linear` and `C=10`. In the end classifier-related
data along with scaler data was stored in `model.sv` file that was later used in pipeline construction.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I ended up using 9 raws of sliding windows. I took into consideration a notion of perspective and searched for faraway cars with smaller windows near the top of part of image that contains road. Then I used larger windows to search for closer cars.
Here is the bunch of images describing all raws of windows that I used. Under each image corresponding parameters `ystrat`, `ystop` and `scale` of raw of windows are situated.

![alt text][image13]

Windows parametrs: `ystart=400`, `ystop=464`, `scale=1`.

![alt text][image14]

Windows parametrs: `ystart=416`, `ystop=480`, `scale=1`.

![alt text][image15]

Windows parametrs: `ystart=400`, `ystop=480`, `scale=1.25`.

![alt text][image16]

Windows parametrs: `ystart=400`, `ystop=496`, `scale=1.5`.

![alt text][image17]

Windows parametrs: `ystart=432`, `ystop=528`, `scale=1.5`.

![alt text][image18]

Windows parametrs: `ystart=432`, `ystop=560`, `scale=2`.

![alt text][image19]

Windows parametrs: `ystart=464`, `ystop=592`, `scale=2`.

![alt text][image20]

Windows parametrs: `ystart=432`, `ystop=624`, `scale=3`.

![alt text][image21]

Windows parametrs: `ystart=400`, `ystop=656`, `scale=4`.

I have already mentioned that my initial pipeline was very slow so one way to make it faster was to use linear kernel for SVM and the other one was to use `pix_per_cell=16` to reduce number of windows in a window raw (it reduces number of `nxblocks` in
`find_cars()` function in `helpers.py` file). I tried to use various combination of windows to reduce false negatives (car was on image but was not found) as much as possible so I ended up with this list.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

