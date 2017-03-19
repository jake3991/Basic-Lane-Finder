# Basic-Lane-Finder
Lane finder with Sklearn 

This project takes a dashcam video feed and highlights the lane lines.  


Reflections
Congratulations on finding the lane lines! As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust? Where will your current algorithm be likely to fail?
Please add your thoughts below, and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!

This code is based on the sklearn linear regression which means its weakeness is curve fitting. If the road were curved or just not straight it would cause an issue. It could plot an aproximation of the curve as a line not actually a curve which would not look to nice!

I also think that an object with a similar slope to the lane lines could cause a problem. For example a jersey barrier in the median could be interpereted as the lane. Maybe with the parameters you suggested it would work or a better tuned in polygon mask. However I am pleased with the output and the methodology.  It gives a very smooth looking line with no noise in the video.  
