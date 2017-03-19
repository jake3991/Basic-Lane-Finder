#import what is needed
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import *
from sklearn import datasets, linear_model
import scipy


#Some functions
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = int(255)
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#pipeline 
def proccess_image(image):
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

	#Apply a gaussian
	kernel_size = 1
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

	#Canny
	low_threshold = 50
	high_threshold = 150
	#50,150
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

	#Change field of view to area with lane
	imshape = image.shape
	vert = np.array([[(0,imshape[0]),(415, 340), (570,340), (imshape[1],imshape[0])]], dtype=np.int32)
	#masked_edges=region_of_interest(edges,vert)
	masked_edges=region_of_interest(edges,vert)

	#Hough Transform Params
	rho = 1 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 25    # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 50 #minimum number of pixels making up a line
	max_line_gap = 200 # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on
	#40,10,50
	#Run Hough Transform on masked section
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)

	#Create Shell lists
	right_x=[]
	right_y=[]
	left_x=[]
	left_y=[]

	# break data out into shell lists and remove outliars based on slope. 
	for line in lines:
		for x1,y1,x2,y2 in line:
			m=(float(y2)-y1)/(x2-x1)
			if m<.75 and m>.45:
				right_x.append(x1)
				right_x.append(x2)
				right_y.append(y1)
				right_y.append(y2)

			elif m<-0.6 and m>-.85:
				left_x.append(x1)
				left_x.append(x2)
				left_y.append(y1)
				left_y.append(y2)
				
	#get the intecept and slope from an sklearn regression for the right line
	right_x=np.transpose(np.matrix(right_x))
	right_y=np.transpose(np.matrix(right_y))
	regr = linear_model.LinearRegression()
	regr.fit(right_x, right_y)
	m_r=regr.coef_
	b_r=regr.intercept_ 
	y_1=540
	y_2=400
	
	#get the points from the line
	x_1=(y_1-b_r)/m_r
	x_2=(y_2-b_r)/m_r

	#draw the line
	cv2.line(line_image,(x_1,y_1),(x_2,y_2),(255,0,0),10)
	
	#same again for the left
	left_x=np.transpose(np.matrix(left_x))
	left_y=np.transpose(np.matrix(left_y))
	regr = linear_model.LinearRegression()
	regr.fit(left_x, left_y)
	m_r=regr.coef_
	b_r=regr.intercept_ 
	y_1=540
	y_2=400

	#get the left points
	x_1=(y_1-b_r)/m_r
	x_2=(y_2-b_r)/m_r

	#Draw the left line
	cv2.line(line_image,(x_1,y_1),(x_2,y_2),(255,0,0),10)	

	#Ovelay the images
	lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0) 

	#return the lane picture 
	return lines_edges


white_output = 'white123.mp4'
clip1 = VideoFileClip("/Users/johnmcconnell/Desktop/CarND-LaneLines-P1/solidWhiteRight.mp4",audio=False)
white_clip = clip1.fl_image(proccess_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


