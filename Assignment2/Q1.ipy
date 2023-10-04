import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def log_kernel(sigma, size):
    size += size % 2 == 0
    s2 = sigma**2
    idx_range = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    x_idx, y_idx = np.meshgrid(idx_range, idx_range)

    tmp_cal = -(x_idx**2 + y_idx**2) / (2.*s2)
    kernel = np.exp(tmp_cal)
    kernel[kernel < np.finfo(float).eps * np.amax(kernel)] = 0

    k_sum = np.sum(kernel)
    if k_sum != 0:
        kernel /= k_sum

    tmp_kernel = kernel * (x_idx**2 + y_idx**2 - 2*s2) / (s2**2)
    kernel = tmp_kernel - np.sum(tmp_kernel) / (size ** 2)
    return kernel


img= cv.imread('the_berry_farms_sunflower_field.jpg', cv.IMREAD_REDUCED_COLOR_4)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0

sigma0 = 0.5
k = np.sqrt(2)
scales = 12
sigmas = sigma0 * k**np.arange(scales)

img_stack = []
for sigma in sigmas:
    size = int(2 * np.ceil(4 * sigma) + 1)  
    kernel = log_kernel(sigma, size) * sigma**2  #LoG filtering
    filtered = cv.filter2D(gray, cv.CV_32F, kernel) 
    filtered = np.power(filtered, 2)
    img_stack.append(filtered)

img_stack = np.dstack(img_stack)

scale_space = []
for i in range(scales):
    filtered = cv.dilate(img_stack[:, :, i], np.ones((3, 3)), cv.CV_32F, (-1, -1), 1, cv.BORDER_CONSTANT)
    scale_space.append(filtered)
scale_space = np.dstack(scale_space)

max_stack = np.max(scale_space, axis=2) 
max_stack = np.repeat(max_stack[:, :, np.newaxis], scales, axis=2) 
max_stack = np.multiply((max_stack == scale_space), scale_space)

radius_vec = []
x_vec = []
y_vec = []

for i in range(scales):
    radius = np.sqrt(2) * sigmas[i]
    threshold = 0.01

    valid = (max_stack[:, :, i] == img_stack[:, :, i]) * img_stack[:, :, i]
    valid[valid <= threshold] = 0

    (x, y) = np.nonzero(valid)

    x_vec.extend(x)
    y_vec.extend(y)
    radius_vec.extend([radius] * len(x))

#Parameters of largest circle
max_radius = np.max(radius_vec)
max_circle_idx = np.argmax(radius_vec)
max_x = x_vec[max_circle_idx]
max_y = y_vec[max_circle_idx]

print()
print(f"Largest Circle Radius: {max_radius}")
print(f"Center Coordinates: ({max_x}, {max_y})")

#Sigma values used
min_sigma = np.min(sigmas)
max_sigma = np.max(sigmas)
print(f"\nRange of sigma values used: [{min_sigma}, {max_sigma}]")

# Draw the circles on the original image
blob_detected_image    = img.copy()  
for i in range(np.size(x_vec)):
        cv.circle(blob_detected_image, (y_vec[i], x_vec[i]), int(radius_vec[i]), (0, 0, 255), 2)  

cv.imwrite("blob_detected_image.jpg", blob_detected_image)


