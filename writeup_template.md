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
[image22]: ./output_images/windows_trouble_YUV1.png
[image23]: ./output_images/windows_HSV1.png
[image24]: ./output_images/windows_HSV2.png
[image25]: ./output_images/windows_HSV3.png
[image26]: ./output_images/windows_HSV4.png
[image27]: ./output_images/windows_HSV5.png
[image28]: ./output_images/windows_HSV6.png
[image29]: ./output_images/heat_map_1.png
[image30]: ./output_images/heat_map_2.png
[image31]: ./output_images/heat_map_3.png
[image32]: ./output_images/heat_map_4.png
[image33]: ./output_images/heat_map_5.png
[image34]: ./output_images/heat_map_6.png
[image35]: ./output_images/heat_threshold_1.png
[image36]: ./output_images/heat_threshold_2.png
[image37]: ./output_images/heat_threshold_3.png
[image38]: ./output_images/heat_threshold_4.png
[image39]: ./output_images/heat_threshold_5.png
[image40]: ./output_images/heat_threshold_6.png
[image41]: ./output_images/labels_1.png
[image42]: ./output_images/labels_2.png
[image43]: ./output_images/labels_3.png
[image44]: ./output_images/labels_4.png
[image45]: ./output_images/labels_5.png
[image46]: ./output_images/labels_6.png


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

First of all I downloaded vehicles and non-vehicles data and stored it `vehicles` and `non-vehicles` directories (I am not providing it as it is too big, thiugh I stored final trained SVM). 3 typical examles of vehicle and non-vehicle are shown below.

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
slower than `linear` kernel and as classifier decision was by far the slowest single operation in pipeline I was forced to move back to `linear` kernel. So my final parameters were `kernel=linear` and `C=10`. In the end classifier-related
data along with scaler data was stored in `model.sv` file that was later used in pipeline construction.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I ended up using 9 raws of sliding windows. I took into consideration a notion of perspective and searched for faraway cars with smaller windows near the top of part of image that contains road. Then I used larger windows to search for closer cars.
Here is the bunch of images describing all raws of windows that I used. Images were made with help of `visualize_windows()` function in `test_utils.oy` file. Under each image corresponding parameters `ystrat`, `ystop` and `scale` of raw of windows are situated.

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

I tried to use various combination of windows to reduce false negatives (car was on image but was not found) as much as possible so I ended up with this list. I noticed that a lot of small windows needed near the top
of search area. On the other hand there was no need in large amount of big windows near the bottom of search area.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

First of all I want to discuss performance of pipline here. My first pipeline implementation based on HOG + color features, and SVM with `rbf` kernel was unacceptibly slow. It took it 7 seconds to work with each frame. I switched to use `pix_per_cell=16` (to reduce ]
number of windows in a window row (it reduces number of `nxblocks` in `find_cars()` function in `helpers.py` file)) and linear kernel. That pipeline worked on 0.5 seconds per frame speed. I removed color features from features vector as it helped me to reduce number
of false negatives. In fact I had to reevaluate my whole feature vector at that point to obtain an acceptable solution. I decided against `HLS` color space as it gave me false negatives. Whole idea behind my pipline was to obtain a solution almost without false 
negatives and then using filtering weight out false positives. That was the reason I dropped `LUV` and `YUV` color spaces as it gave me too much false positives on lane lines to weight it out without losing cars. Typical situation (before filtering) is on image below.

![alt text][image22]

So I chose `HSV` color space for my pipeline. Now let us discuss it in more details using test images. Images were generated with help of `test_find_cars()` function in `test_utils.py` file.
First step was to find all windows that presumably contain car using SVM and sliding window approach. Results shown on images below.

![alt text][image23] 
![alt text][image24]
![alt text][image25]
![alt text][image26]
![alt text][image27]
![alt text][image28]

As we can see there are some false positives (which is expectable as main idea was to minimize false negatives) but most of them are easy to weight out except for one in the shadows on fourth image. Now lets look at heat map of these images.

![alt text][image29]
![alt text][image30]
![alt text][image31]
![alt text][image32]
![alt text][image33]
![alt text][image34]

Next step is to apply heat threshold and remove parts of images that are not hot enought (here we set threshold to 2).

![alt text][image35]
![alt text][image36]
![alt text][image37]
![alt text][image38]
![alt text][image39]
![alt text][image40]

As it can be seen there are false positives still. Next step is to draw a rectangle around each hot object using `scipy.ndimage.measurements.label()` function.

![alt text][image41]
![alt text][image42]
![alt text][image43]
![alt text][image44]
![alt text][image45]
![alt text][image46]

That is the end of pipeline. As we can see there are some false positives but we can address it during pipeline application to video file. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Resulting video can be found in `output_project_video.mp4` file.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As it was mentioned at the end of pipeline section there were false positives that needed to be weighted out. To mitigate that problem I implemented `RingBuffer` class. It can be found in `helpers.py` file. It stores data from detections on previous 7 frames. I tuned it
so for rectangle to appear it should be detected 17 times during current frame and last 7 frames combined (`make_pipeline()` function in `helpers.py` file). That approach allowed me to exclude false positives from video although cars a tracked on almost every frame. That approach was based at the idea that car was 
moving in a relatively slow manner across the image so it allowed me to use some memory of its locations. On the other hand, false positives appeared in different parts of image so it were weighted out by that approach.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem I faced in this project is to make my ppipeline fast enough so it can actually process a video in a resonable time. After I profiled my pipeline and found its weak spots I was forced to refactor it from top to bottom using faster but less precise
approach. And still my approach is not real-time at all as it takes my pipeline 12 seconds to process one second of video. Second problem was that I preferd false positives to false negatives so I ended up with pipeline that should use 7 frame memory to weight out false
positives. Maybe if I was more tolerant to false negatives and tried to balance in with false positives it would result in less memory frame that as a consequence would allow me to track cars in the image earlier. Third problem is that my pipeline sees two cars that are
near each other as one big car. It is expected as I am not tracking a car in an intelligent manner just as a hot region on image. Maybe If I use some intelligent tracking method it would allow me to avoid using searching window method on each video frame that in turn
would speed up my pipeline. Fourth problem is SVM + HOG approach itself. It has 98-99% accuracy on data sets yet there are a lot of false positives and false negatives during window search. It took me a lot of time to assemble all those peaces in an acceptable pipeline.
Maybe an approach using convolutional neural network trained on cars and non-cars data sets can help improve accuracy. On the other hand, if it makes execution time of pipeline much worse then it can not be used here.
