import cv2 as cv
import numpy as np

#function to obtain coordinates of points clicked on the image
n = 0
points = np.empty((4,2))

def click_event(event, x, y, flags, params):
   global n

   if event == cv.EVENT_LBUTTONDOWN:
      points[n] = [x,y]      
      
      cv.circle(img, (x,y), 3, (0,255,255), -1) #draw point on the image
      n += 1
 
img = cv.imread('fort.jpg')

cv.namedWindow('Point Coordinates')
cv.setMouseCallback('Point Coordinates', click_event)

while True:
   cv.imshow('Point Coordinates',img)
   k = cv.waitKey(1) & 0xFF
   if k == 27:
      break
cv.destroyAllWindows()

flg = cv.imread('flag.jpg')
flg_points = np.array([[0, 0], [flg.shape[1], 0], [flg.shape[1], flg.shape[0]], [0, flg.shape[0]]], dtype=np.float32)

#Homography matrix
homography_matrix, _ = cv.findHomography(flg_points, points)

#Image warping
warped = cv.warpPerspective(flg, homography_matrix, (img.shape[1], img.shape[0]))

output = cv.addWeighted(img, 1, warped, 0.9, 0)

cv.imshow('Superimposed Image', output)
cv.waitKey(0)
cv.destroyAllWindows()
