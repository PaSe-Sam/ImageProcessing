from scipy.optimize import minimize
from scipy import linalg
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np

def line_param(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m*x1

    a = m
    b = -1
    d = -c

    magnitude = (a**2 + b**2)**0.5

    a_normalized = a / magnitude
    b_normalized = b / magnitude
    d_normalized = d / magnitude

    return a_normalized, b_normalized, d_normalized
def RANSAC_line(X, iterations, threshold, min_inliers):

    best_model = None
    best_inliers = []

    for _ in range(iterations):
        sample_indices = np.random.choice(len(X), 2, replace=False)
        x1, y1 = X[sample_indices[0]]
        x2, y2 = X[sample_indices[1]]
        
        a, b, d = line_param(x1, y1, x2, y2)

        distances = np.abs(a*X[:,0] + b*X[:,1] - d)
        inliers = np.where(distances < threshold)[0]

        if len(inliers) >= min_inliers and len(inliers) > len(best_inliers):
            best_model = (a, b, d)
            best_inliers = inliers

    return best_model, best_inliers

def circle_param(x1, y1, x2, y2, x3, y3):
    mx1, my1 = (x1 + x2) / 2, (y1 + y2) / 2
    mx2, my2 = (x2 + x3) / 2, (y2 + y3) / 2

    gradient1 = 0 if y2 - y1 == 0 else (x2 - x1) / (y2 - y1)
    gradient2 = 0 if y3 - y2 == 0 else (x3 - x2) / (y3 - y2)

    xc = (gradient1 * mx1 - gradient2*mx2 + my2 - my1) / (gradient1 - gradient2)
    yc = -gradient1 * (xc - mx1) + my1

    r = ((x1 - xc)**2 + (y1 - yc)**2)**0.5

    return xc, yc, r

def RANSAC_circle(X, iterations, threshold, min_inliers):
    
    best_model = []
    best_inliers = []
    
    for _ in range(iterations):
        sample_indices = np.random.choice(len(X), 3, replace=False)
        x1, y1 = X[sample_indices[0]]
        x2, y2 = X[sample_indices[1]]
        x3, y3 = X[sample_indices[2]]
        
        xc, yc, r = circle_param(x1, y1, x2, y2, x3, y3)
        
        errors = np.abs(np.sqrt((X[:,0] - xc)**2 + (X[:,1] - yc)**2) - r)
        
        inliers = np.where(errors < threshold)[0]
        
        if len(inliers) >= min_inliers and len(inliers) > len(best_inliers):
            best_model = (xc, yc, r)
            best_inliers = inliers
                
    return best_model, best_inliers

# Generate noisy data points corresponding to a circle and a line
N = 100
half_n = N//2

r = 10
x0_gt, y0_gt = 2,3  # Center 

s = r/16
t = np.random.uniform(0,2*np.pi,half_n)
n = s * np.random.randn(half_n)
x = x0_gt+(r+n)*np.cos(t)
y = y0_gt+(r+n)*np.sin(t)

X_circ = np.column_stack((x, y))

s = 1.0
m,b = -1,2
x = np.linspace(-12,12,half_n)
y = m*x+b+s*np.random.randn(half_n)

X_line = np.column_stack((x,y))

X = np.vstack((X_circ,X_line))
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_line[:, 0], X_line[:, 1], label='Line')
ax.scatter(X_circ[:, 0], X_circ[:, 1], label='Circle')

circle_gt = plt.Circle((x0_gt, y0_gt), r, color='g', fill=False, label='Ground truth circle')
ax.add_patch(circle_gt)
ax.plot(x0_gt, y0_gt, '+', color='g')

x_min, x_max = ax.get_xlim()
x_ = np.array([x_min, x_max])
y_ = m * x_ + b
plt.plot(x_, y_, color='m', label='Ground truth line')

#RANSAC parameters
iterations  = 100000
threshold   = 0.2
min_inliers = 15

#Line estimation
best_RANSAC_line,line_inlier_indices_array = RANSAC_line(X_line,iterations,threshold,min_inliers)

remnant_indices = [i for i in range(len(X)) if i not in line_inlier_indices_array]
remnant_points  = X[remnant_indices]

#Circle estimation
best_RANSAC_circle, circle_inlier_indices_array = RANSAC_circle(remnant_points,iterations,threshold,min_inliers)

x_min,x_max = ax.get_xlim()
x_ = np.array([x_min,x_max])
y_ = (-best_RANSAC_line[0]*x_+best_RANSAC_line[2])/best_RANSAC_line[1]
plt.plot(x_,y_,label='RANSAC line',color='black')
ax.scatter(X_line[line_inlier_indices_array, 0],X_line[line_inlier_indices_array, 1],color='blue',label='Line Inliers')

best_line_sample_indices = line_inlier_indices_array[:2]
best_line_samples = X_line[best_line_sample_indices]
ax.scatter(remnant_points[circle_inlier_indices_array,0],remnant_points[circle_inlier_indices_array,1],color='green',label='Circle Inliers')

x_center,y_center,radius = best_RANSAC_circle
circle_estimated = plt.Circle((x_center,y_center),radius,color='purple',fill=False,label='RANSAC Circle')
ax.add_patch(circle_estimated)


best_circle_sample_indices = circle_inlier_indices_array[:3]
best_circle_samples = remnant_points[best_circle_sample_indices]

ax.scatter(best_line_samples[:,0], best_line_samples[:,1],color='red',marker='^',label='Best 2 Samples for Line')
ax.scatter(best_circle_samples[:,0], best_circle_samples[:,1],color='orange',marker='v',label='Best 3 Samples for Circle')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
