import os
import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
from moviepy.editor import VideoFileClip



class Lane():


    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.frame = 0
        self.left_fit = None
        self.right_fit = None
        self.ploty = None
        self.last_left_fit = None
        self.last_right_fit = None


        self.debug_frame = False
 

    def process_video(self, source_fname, output_fname):

        path = os.path.dirname(os.path.abspath(__file__)) + '\\' + source_fname

        if not os.path.exists(path):
            print('Error: wrong path: ', path)
            return

        try:
            clip2 = VideoFileClip(path)#.subclip(0,3)  #20,49
        except IOError:
            print("Something went wrong with video processing\n\r")

        video_clip = clip2.fl_image(self.process_frame)
        video_clip.write_videofile(output_fname, audio=False)

        return

    def process_frame(self, source_img):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Create undistorted image
        undistorted_img = cv2.undistort(
            source_img, self.mtx, self.dist, None, self.mtx)

        persp_img, M2, Minv = self.perspective_transform(undistorted_img)

        # Create combined tresholds image 
        # correct for image processing 

        if test_images:
            combined = self.combined_thresholds(ksize=3, image=persp_img[:,:,::-1])
        else:
            # for video
            combined = self.combined_thresholds(ksize=3, image=persp_img)
 
 
 
        # Check if exist previous poly and choose the method to proceed
        if self.left_fit is None or self.right_fit is None:
            self.left_fit, self.right_fit = self.find_lane_pixels(combined)
        else:
            self.left_fit, self.right_fit = self.search_around_poly(
                combined,  self.left_fit, self.right_fit)

        # Measure real curvature
        left_curve, right_curve = self.measure_curvature_real(
            self.left_fit, self.right_fit, self.ploty)
        curve_average = np.average([left_curve, right_curve])

        out_img = self.draw_lane(undistorted_img, Minv, self.left_fit, self.right_fit)

        # Print radius and car position
        radius = "Radius of curvature: {0} m".format(int(curve_average))
        cv2.putText(out_img, radius, (50, 50), font, 1.2, (255, 255, 255), 2)
        side, diff = self.find_car_position(combined.shape ,self.left_fit, self.right_fit)
        diff_car = "Car position: {0:1.2f} m {1} of center".format(  diff, side )
        cv2.putText(out_img,diff_car,(50,100), font, 1.2,(255,255,255),2) 
    


        # draw test lines
    
       
        if draw_lines:
            radius = "frame: {0}".format( self.frame)
            cv2.putText(out_img, radius, (750, 50), font, 1.2, (255, 255, 255), 2)
            ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.transpose(np.vstack([right_fitx , ploty]))])

            combined_screen = np.dstack((combined,combined,combined)) *255
            # Dadd computed polylines
            cv2.polylines(combined_screen, np.int_([pts_left]), False, (255,0,0),3)
            cv2.polylines(combined_screen, np.int_([pts_right]), False, (0,255,0),3)
  
 
    
            winname = 'Result'
            cv2.namedWindow(winname)
            cv2.imshow(winname, out_img[:,:,::-1])
            cv2.waitKey(5)


        # fn = 'my_test_out2/video' + str(self.frame) + '.jpg'
        # cv2.imwrite(fn, out_img[:,:,::-1])
        # fn = 'my_test_out2/video_source_' + str(self.frame) + '.jpg'
        # cv2.imwrite(fn, source_img[:,:,::-1])


        self.frame +=1
        return out_img
 
    
    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # Find the peak of the left and right halves of the histogram

        # # Visualize  histogram
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 20))
        # ax1.imshow(binary_warped, cmap="gray")
        # ax1.set_title('Binary warped image',  fontsize=10)
        # ax2.set_title('Histogram', fontsize=10)
        # plt.plot(histogram)
        # plt.xlim(0, 1280)
        # plt.ylim(0, 360)
        # plt.pause(5)
 
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin  
            win_xleft_high = leftx_current + margin 
            win_xright_low = rightx_current - margin   
            win_xright_high =rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

 

        self.last_left_fit, self.last_right_fit, self.ploty = self.fit_polynomial(binary_warped.shape, leftx, lefty, rightx, righty)

        # ## Visualization ##
        # out_img[lefty, leftx, 0] = 255# [255, 0, 0]
        # out_img[righty, rightx,1] = 255#[0, 0, 255]

        # ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
        # left_fitx = self.last_left_fit[0]*ploty**2 + self.last_left_fit[1]*ploty + self.last_left_fit[2]
        # right_fitx =self.last_right_fit[0]*ploty**2 + self.last_right_fit[1]*ploty + self.last_right_fit[2]
        # pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        # pts_right = np.array([np.transpose(np.vstack([right_fitx , ploty]))])

        
        # # Dadd computed polylines
        # cv2.polylines(out_img, np.int_([pts_left]), False, (255,0,0),3)
        # cv2.polylines(out_img, np.int_([pts_right]), False, (0,255,0),3)
        # winname = 'Result'
        # cv2.namedWindow(winname)
        # cv2.imshow(winname, out_img[:,:,::-1])
        
        # fn = 'my_test_out/sliding_window.jpg'
        # cv2.imwrite(fn, out_img[:,:,::-1])
        # cv2.waitKey(5000)


        return self.last_left_fit, self.last_right_fit

    def search_around_poly(self, binary_warped, left_fit, right_fit):
  
       
        # Fast search pixels around the previously found polynomial 
 
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
 
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + margin)))
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]


        left_fitx, right_fitx, ploty =  self.fit_polynomial(binary_warped.shape, leftx, lefty, rightx, righty) 

        left_curve, right_curve = self.measure_curvature_real( left_fitx, right_fitx, self.ploty )
         
        self.last_left_fit =  left_fitx 
        self.last_right_fit = right_fitx
 
        return self.last_left_fit,  self.last_right_fit
    
    def fit_polynomial(self, binary_shape,leftx, lefty, rightx, righty):
 
        # Fit polynomial
        left_fit = np.polyfit(lefty,leftx,deg=2)
        right_fit = np.polyfit(righty,rightx,deg=2)
    
        if self.ploty is None:
            self.ploty = np.linspace(0, binary_shape[0]-1, binary_shape[0] )

        left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]


        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx , self.ploty])))])
 
 
        return left_fit, right_fit, self.ploty



    def color_binary_pipeline(self, img, s_thresh=(225, 240), sx_thresh=(20, 100), r_tresh=(125, 255)):
        
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1] # L channel from HLS
        s_channel = hls[:, :, 2] # S channel from HLS
    
        r_channel = img[:,:,0]  # Red channel from RGB


        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) &
                (scaled_sobel <= sx_thresh[1])] = 1
        sxbinary[img.shape[0]-50:, :] = 0  #remove car hood pixels

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        s_binary[img.shape[0]-150:, :] = 0  #remove car hood pixels

        # Isolate yellow color from L channel of HLS image
        hls_yellow = np.zeros_like(hls[:,:,0])
        hls_yellow[((hls[:,:,0] >= 15) & (hls[:,:,0] <= 35))
                    & ((hls[:,:,1] >= 30) & (hls[:,:,1] <= 205))
                    & ((hls[:,:,2] >= 115) & (hls[:,:,2] <= 255))                
                    ] = 1
        # Yellow from R channel of RGB
        r_binary = np.zeros_like(r_channel)
        r_binary[(r_channel >= r_tresh[0]) & (r_channel <= r_tresh[1])] = 1
        r_binary[img.shape[0]-30:, :] = 0
    
        return sxbinary, s_binary, hls_yellow, r_binary
    

    def combined_thresholds(self, ksize, image):
 
        abs_thresh=(15, 100)
        mag_threshold=(50, 170)
        dir_bin_thresh=(1.2, 1.3)
        s_thresh=(220, 250)
        sx_thresh=(20, 100)
        r_thresh=(215, 255)
    
    
        # Thresholded absolute value of sobel operator
        gradx = abs_sobel_thresh(image, 'x', ksize, abs_thresh)
        grady = abs_sobel_thresh(image, 'y', ksize, abs_thresh)

        # Thresholded magnitude of the gradient
        mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=mag_threshold)
        # Thresholded gradient direction
        dir_binary = dir_threshold(image, sobel_kernel=15, thresh=dir_bin_thresh )

        # Color gradient, RED  and yellow color from HLS
        (sxbinary, s_binary, hls_yellow, r_binary) = self.color_binary_pipeline(image, s_thresh, sx_thresh, r_thresh)

  
        combined = np.zeros_like(dir_binary)
    
        # Combining all Thresholds
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) |(sxbinary == 1) | (r_binary == 1) |  (hls_yellow == 1)  ] = 1
         #  Not used binary as not useful for the task:  (b_binary == 1) (l_binary == 1) |  (s_binary == 1) 
    
 

        if show_combined_thresholds:
            #Plot different tresholds
            grad = np.zeros_like(dir_binary)
            mg = np.zeros_like(dir_binary)
            grad[(gradx == 1) & (grady == 1)] = 1
            mg[(mag_binary == 1) & (dir_binary == 1)] = 1

            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(40,20))
            ax1.imshow(image , cmap="gray")
            ax1.set_title('Source image', fontsize=10)

            ax2.imshow(grad , cmap="gray")
            ax2.set_title('Gradient X & Y', fontsize=10)

            ax3.imshow(sxbinary, cmap="gray")
            ax3.set_title('SX threshold', fontsize=10)

            ax4.imshow(mag_binary, cmap="gray")
            ax4.set_title('Thresholded magnitude', fontsize=10)


            f, (bx1, bx2, bx3) = plt.subplots(1, 3, figsize=(40,20))
            bx1.imshow(r_binary, cmap="gray")
            bx1.set_title('Thresholded R channel', fontsize=10)

            bx2.imshow(hls_yellow, cmap="gray")
            bx2.set_title('Tresholded yellow', fontsize=10)
 
            bx3.imshow(combined, cmap="gray")
            bx3.set_title('Combined threshold', fontsize=10)
            plt.pause(5)


        return combined

     

        
    def find_car_position(self, image_shape, left_best_fitx, right_best_fitx):
        # Find the position of the car from the lane center
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension    
        image_center = image_shape[1]/2 

        y = 700
        
        
        left_line = left_best_fitx[0]*(y**2) + left_best_fitx[1]*y + left_best_fitx[2]
        right_line = right_best_fitx[0]*(y**2) + right_best_fitx[1]*y + right_best_fitx[2]
        lane_middle =  (right_line - left_line)/2. + left_line  
    
        diff_from_center = (image_center - lane_middle) * xm_per_pix
        # Check if the car is left or right from the center
        if (diff_from_center > 0):
            side = "right"
        else:
            side = "left"
    
        return side, np.abs(diff_from_center)



    def measure_curvature_real(self, left_fit_cr, right_fit_cr,ploty ):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
        y_eval = np.max(ploty)
        
        #Calculation of R_curve (radius of curvature) 
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        
        return left_curverad, right_curverad



                    
    def perspective_transform(self, image):

        image_width = image.shape[1]
        image_height = image.shape[0]

        # test5 ok 70 120 90 120
        x_offset_top = 70
        x_offset_bottom = 120
        y_offset = 90
        cam_x_offset = 100
        # Compute source and target points
        src = np.float32([[x_offset_bottom, image_height], [image_width/2 - x_offset_top, image_height/2 + y_offset], [image_width/2 + x_offset_top, image_height/2 + y_offset],[image_width - x_offset_bottom, image_height]])
        dst = np.float32([[x_offset_bottom + cam_x_offset, image_height], [x_offset_bottom, 0], [image_width - x_offset_bottom, 0], [image_width - x_offset_bottom - cam_x_offset, image_height]])
    
        # Compute M and Minv for warpPerspective
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(image, M, (image_width, image_height), flags=cv2.INTER_LINEAR)

        return warped, M, Minv


        

    def draw_lane(self, source_image, Minv, left_fit,right_fit):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Generate x and y values for plotting
        width = source_image.shape[1]
        height = source_image.shape[0]

        ploty = np.linspace(0, source_image.shape[0]-1, source_image.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        # Create an image to draw the lines on

        warp_zero = np.zeros_like(source_image[:,:,0]).astype(np.uint8)
    
        warped = np.dstack((warp_zero, warp_zero, warp_zero))
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx , ploty])))])
        pts = np.hstack((pts_left, pts_right))
            


        # Draw the lane onto the warped blank image
        cv2.fillPoly(warped, np.int_([pts]), (0, 255, 0))
        # Unwarp the image with plotted lane
        img_size = (width,height)
        unwarped = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)
    
        result = cv2.addWeighted(source_image, .7, unwarped, 0.3, 0)

        
        return result


def calibration(imgfolder, nx, ny):
 
    # Calibrate camera
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(imgfolder)

    # Step through the list and search for chessboard corners

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
 
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Test undistortion on an image
    img = cv2.imread('camera_cal/calibration2.jpg')
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('output_images/test_undist_cal2.jpg', dst)

    # Save the camera calibration result for later use 
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("output_images/wide_dist_pickle.p", "wb"))

    # Visualize undistortion
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=30)
    # ax2.imshow(dst)
    # ax2.set_title('Undistorted Image', fontsize=30)
    # plt.pause(5)
    return mtx, dist

 

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(25, 200)):

    # Compute absolute value of the derivative
 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(100, 255)):

    # Compute mag thresholds 
 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobx**2 + soby**2)
    scale = np.max(mag)/255
    mag = np.uint8((mag/scale))
    binary_output = np.zeros_like(mag)
    binary_output[(mag >= mag_thresh[0]) & (mag <= mag_thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Direction thresholds
 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output
 



 

# main -------------------------------
 
# 1. calibrate camera
# folder with calibration images
imgfolder = 'camera_cal/calibration*.jpg'
nx = 9  # the number of inside corners in x
ny = 6  # the number of inside corners in y
mtx, dist = calibration(imgfolder, nx, ny)

# -------------------------------------------------------------------
# 2. undistort image
# loading mtx and dist (camera matrix and distortion coefficients)
dist_pickle = pickle.load(open("output_images/wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# -------------------------------------------------------------------


video =  True   # True to process project_video.mp4
video2 =   False  # True to process challenge_video.mp4

show_combined_thresholds = False # True to show image with combined thresholds
debug = False
draw_lines =  False   # True to result image while generation video

lane = Lane(mtx,dist)  # Create Lane object and pass mtx,dist for correction of image distortion


# -------------------------------------------------------------------
# Process test images from test_images frolder
test_images = False
 
if test_images:
    images = glob.glob('test_images/*')

    
    draw_lines = False
    show_tershold = False
    show_rlb_tershold = False
    for idx, fname in enumerate(images):
        print('Processing: ',fname)
        img = cv2.imread(fname)
        out_img = lane.process_frame(img)
        fn = 'output_images/test_images/test_out_' + str(idx) + '.jpg'
        cv2.imwrite(fn, out_img)

 


# -------------------------------------------------------------------
# Process image
 
source_img = cv2.imread('test_images/test5.jpg')   

if not video:
    out_img = lane.process_frame(source_img)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(source_img[:,:,::-1], cmap="gray")
    ax1.set_title('Original', fontsize=30)
    ax2.imshow(out_img[:,:,::-1], cmap="gray")
    ax2.set_title('Identified lane', fontsize=30)
    plt.pause(10)
 

# -------------------------------------------------------------------
# Process video
 
if video:
    draw_lines = True
    show_tershold =   False #True 
    show_rlb_tershold = False #
    lane.process_video('project_video.mp4', 'project_output.mp4')


if video2:
    draw_lines = True
    show_tershold =   False #True 
    show_rlb_tershold = False #
    lane.process_video('challenge_video.mp4', 'challenge_out.mp4')


 
