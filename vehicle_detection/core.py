
# import cv2
from moviepy.editor import VideoFileClip

import helpers
# import test_utils


def run():
    """
    @brief main entry point uses pipeline to process given video
    """
    # test_utils.visualize_hog()
    # test_utils.visualize_windows()

    # test_utils.test_data_prep('./../vehicles', './../non-vehicles', 'HSV')
    # test_utils.test_features_prep('./../vehicles', './../non-vehicles')
    # test_utils.test_find_cars('./../test_images', 'HSV')

    ret, mtx, dist = helpers.calibrateCamera('./../camera_cal/')
    # test_utils.test_camera_calibration('./../camera_cal/', mtx, dist)

    pipeline = helpers.make_pipeline(mtx, dist, 'HSV')

    output_file = './../output_project_video.mp4'
    clip1 = VideoFileClip('./../project_video.mp4')
    # clip1.save_frame('./7.0.png', 7.0)
    # clip1 = VideoFileClip('./../project_video.mp4').subclip(20,35)
    output_clip = clip1.fl_image(pipeline)
    output_clip.write_videofile(output_file, audio=False)


if __name__ == '__main__':
    run()
