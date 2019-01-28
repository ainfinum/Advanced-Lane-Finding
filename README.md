## Advanced Lane Finding Project


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1.jpg "Original"
[image2]: ./output_images/test_undist_cal1.jpg "Undistorted"
[image3]: ./output_images/bridge_shadow1.jpg "Original"
[image4]: ./output_images/bridge_shadow1_undistorted.jpg "Undistorted"
[image5]: ./output_images/warped_image3.png "Warped Perspective Ex. 1"
[image6]: ./output_images/warped_image1.png "Warped Perspective Ex. 2"
[image7]: ./output_images/warped_tresholded.png "Warped Perspective Tresholded"
[image8]: ./output_images/thresholded_2_1.jpg "Binary tresholded images example 1"
[image9]: ./output_images/thresholded_2_2.jpg "Binary tresholded images example 1"
[image10]: ./output_images/thresholded_3_1.jpg "Binary tresholded images example 2"
[image11]: ./output_images/thresholded_3_2.jpg "Binary tresholded images example 2"
[image10]: ./output_images/thresholded_4_1.jpg "Binary tresholded images example 3"
[image11]: ./output_images/thresholded_4_2.jpg "Binary tresholded images example 3"
[image12]: ./output_images/histogram.png "Histogram"
[image13]: ./output_images/sliding_window.jpg "Sliding window method"
[image14]: ./output_images/search_around_poly.png "Search around polynomial"

[image15]: ./output_images/test_images/bridge_shadow1.jpg "Result image 1"
[image16]: ./output_images/test_images/straight_lines1.jpg "Result image 2"
[image17]: ./output_images/test_images/straight_lines2.jpg "Result image 3"
[image18]: ./output_images/test_images/test1.jpg "Result image 4"
[image19]: ./output_images/test_images/test2.jpg "Result image 5"
[image20]: ./output_images/test_images/test3.jpg "Result image 6"
[image21]: ./output_images/test_images/test4.jpg "Result image 7"
[image22]: ./output_images/test_images/test5.jpg "Result image 8"
[image23]: ./output_images/test_images/test6.jpg "Result image 9"


[video1]: ./output_images/project_output_ready.mp4 "Video"


## 1. Camera Calibration
#### Compute the camera calibration matrix and distortion coefficients


In order to find the camera calibration matrix and distortion coefficients I created `calibration(imgfolder, nx, ny)` function ( lines #517-567 file `advanced-lane.py`). Where nx and ny are numbers of chessboard corners. 

I loop through all the calibration images to find chessboard corners using `cv2.findChessboardCorners()` function. 
If corners were found I added them to the list and proceed with the next image.
After all images has been processed I used `cv2.calibrateCamera()` function (line #548) to find the camera calibration and distortion coefficients.
I saved calibration matrix and distortion coefficients ( file `output_images/wide_dist_pickle.p`) in order to use it in the future.

Using the `cv2.undistort()` function I obtained the following result: 

![alt text][image1]

![alt text][image2]

 
## 2. Apply a distortion correction to raw camera images.

In this step I'd like to show how distortion correction works with the raw images:

![alt text][image3]

![alt text][image4]

## 3. Perspective transform.

As a useful information for lane finding is only in the bottom half of the image I cut a road lane image area and transform perspective to get a top view of the road.  
To trasform image perspective to the top view of the road I created `perspective_transform()` function  ( lines #452-471 file `advanced-lane.py`)
The the source and destination points are computed from image size and offsets for X and Y 


```python
    x_offset_top = 70
    x_offset_bottom = 120
    y_offset = 90
    cam_x_offset = 100
   
    src = np.float32([[x_offset_bottom, image_height], [image_width/2 - x_offset_top, image_height/2 + y_offset], [image_width/2 + x_offset_top, image_height/2 + y_offset],[image_width - x_offset_bottom, image_height]])
    dst = np.float32([[x_offset_bottom + cam_x_offset, image_height], [x_offset_bottom, 0], [image_width - x_offset_bottom, 0], [image_width - x_offset_bottom - cam_x_offset, image_height]])
 
```
 

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 120,  720     | 220,  720     | 
| 570,  450     | 120,  0       |
| 710,  450     | 1160, 0       |
| 1160, 720     | 1060, 720     |

Using cv2.warpPerspective() function I computed the warped images.
Below are examples of the warped images showing that lines appears parallel:


![alt text][image5]

![alt text][image6]



## 4. Color transforms, gradients methods to create a thresholded binary image.

I used a combination of color and gradient thresholds (L and S channel from HLS and Red channel from RGB images ) to generate binary images ( `color_binary_pipeline` lines #298-336, file `advanced-lane.py`). I applied the Sobel operator to the lightness value of the HLS image `color_binary_pipeline()` ( lines #309-312 file `advanced-lane.py`). I used a combination of the magnitude of the gradient and the direction of the gradient. I isolated yellow and white colors from the original image converted to HSV color space ( lines #301, #326-330 file `advanced-lane.py`) 
The example of my output:

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]


## 4. Identification of lane-line pixels and fit their positions with a polynomial

I created a histogram from the bottom half of the warped binary image to identify left and right peaks. X-position's of these peaks are my starting points to search left and right lines. 

![alt text][image12]

In order to find left and right lines I used the sliding window method `find_lane_pixels()` (lines #126-239) moving upward from starting point. I searched pixels within the window of 100x80 pixels. If minimum of 50 pixels were found I recenter search window based on the mean position of these pixels (lines #170-197).

In `fit_polynomial()` function (lines #276-294) I used Numpy function np.polyfit() fit pixel positions with a polynomial (lines #285-286)

![alt text][image13]

Once the first polynomial has been found I used it for the fast line search method for the next frame. In `search_around_poly()` function (lines #241-274) I just search in a margin around the previous line position.

![alt text][image14]




## 5. Clculate the radius of curvature of the lane and the position of the vehicle with respect to center.

Using and the following equation R_curve = ((1+(2Ay+B)^2)^1.5)/|2A| 

I calculated the radius of curvature in `measure_curvature_real()` function (lines #429-446). In order to convert polynomial curvature to real values in meters I used the following  coefficients:
 
 ym_per_pix = 30/720 # meters per pixel in y dimension

 xm_per_pix = 3.7/700 # meters per pixel in x dimension
 
Calculation of the position of the vehicle with respect to center implemented in the `find_car_position()` function (lines #431-447)



## 6. Final result. Identified lane plotted back down onto the road.

Finaly, the function `draw_lane()` (file `advanced-lane.py`, lines #476-514) plotting identified lane back down onto the road.
As I now have the lane line curves parameters I can use the fillPoly() function (line #506) to draw the lane region. 
I use warpPerspective() function to trasform lane image perspective back to the original image perspective (line #509). 
And the final step is to combine original image with with filled area using addWeighted() function
 
Here are examples of my result:


![alt text][image15]

![alt text][image16]

![alt text][image18]

![alt text][image18]

![alt text][image19]

![alt text][image20]

![alt text][image21]

![alt text][image22]

![alt text][image23]


---
 
# Pipeline (video)

## 1. Final video output.  

Here's a [link to my video result](./output_images/project_output_ready.mp4)

---

### Discussion

#### 1. Problems / issues faced in implementation of this project.  Where the pipeline will likely fail?  What could we do to make it more robust?


I faced issues with brights, shadows and the light-gray bridge sections. I spent some time to figure out better tresholds values. Also I found that several channels produced too much noise for the binary image, so I decided to remove them from the combined tresholds. Generally speaking the various gradient and color thresholds only work in a small set of conditions. I think HD video quality can produce much better result. The current implementaion fail on the challenge videos. I tried to change contrast and brightness of the image but it didn't help a lot.

The pipeline might fail on dark or bright parts of the road. Dirty road, heavy rainfall or snow will add a lot of noise to the binary image and it will be hard to correctly identify lane. Cars on the same lane or crossing the line will create a problem as well.

The following steps can be done to improve the pipeline:

1. HD or better quality video
2. Use color correction filters
3. Monitor of channels activity. Dynamically remove channel data if it produce to much noise (high pixel value)
4. Roll back to previous detected polynomial if new fits are rejected
5. Polynomial comparision function. Comparision left and right polynomial fits with previous values
5. Lane detection with deep learning

 
