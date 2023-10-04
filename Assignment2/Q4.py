import cv2 as cv    
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

img1 = cv.imread('img1.ppm')  
img5 = cv.imread('img5.ppm')

img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img5 = cv.cvtColor(img5, cv.COLOR_BGR2RGB)

sift = cv.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img5, None)

bf = cv.BFMatcher()

matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.65*n.distance:
        good_matches.append(m)

matching_result = cv.drawMatches(img1, keypoints1, img5, keypoints2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(matching_result)
ax.set_xticks([])
ax.set_yticks([])

plt.show()

sift = cv.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img5, None)

bf = cv.BFMatcher()

matches = bf.knnMatch(descriptors1, descriptors2, k=2)
good_matches = []
pt1 = []
pt2 = []

for m, n in matches:
    if m.distance < 0.65*n.distance:
        good_matches.append([m])
        pt1.append(keypoints1[m.queryIdx].pt)
        pt2.append(keypoints2[m.trainIdx].pt)

good_matches, pt1, pt2 = np.array(good_matches), np.array(pt1), np.array(pt2)
matched_img = cv.drawMatchesKnn(img1, keypoints1, img5, keypoints2, good_matches[:50], None, flags=2)

def homography(pt1, pt2):
    mean1, mean2 = np.mean(pt1, axis=0), np.mean(pt2, axis=0)
    s1, s2 = len(pt1)*np.sqrt(2)/np.sum(np.sqrt(np.sum((pt1-mean1)**2, axis=1))), len(pt1)*np.sqrt(2)/np.sum(np.sqrt(np.sum((pt2-mean2)**2, axis=1)))
    tx1, ty1, tx2, ty2 = -s1*mean1[0], -s1*mean1[1], -s2*mean2[0], -s2*mean2[1]
    T1, T2 = np.array(((s1, 0, tx1), (0, s1, ty1), (0, 0, 1))), np.array(((s2, 0, tx2), (0, s2, ty2), (0, 0, 1)))
    A = []

    for i in range(len(pt1)):
        X11, X21 = T1 @ np.concatenate((pt1[i], [1])).reshape(3, 1), T2 @ np.concatenate((pt2[i], [1])).reshape(3, 1)
        A.append((-X11[0][0], -X11[1][0], -1, 0, 0, 0, X21[0][0]*X11[0][0], X21[0][0]*X11[1][0], X21[0][0]))
        A.append((0, 0, 0, -X11[0][0], -X11[1][0], -1, X21[1][0]*X11[0][0], X21[1][0]*X11[1][0], X21[1][0]))
    
    A = np.array(A)
    U, S, V = np.linalg.svd(A, full_matrices=True)
    h = np.reshape(V[-1], (3, 3))
    H = linalg.inv(T2)@h@T1
    H = (1 / H.item(8))*H
    return H
        
def d(P1, P2, H):
    p1 = np.array([P1[0], P1[1], 1])
    p2 = np.array([P2[0], P2[1], 1])

    p2_estimate = np.dot(H, p1.T)
    p2_estimate = (1 / p2_estimate[2])*p2_estimate
    return np.linalg.norm(p2.T - p2_estimate)

def RANSAC(points1, points2):
    inlier_count, selected_inliers = 0, None
    points = np.hstack((points1, points2))
    num_iterations = int(np.log(1 - 0.95)/np.log(1 - (1 - 0.5)**4))
    
    for _ in range(num_iterations):
        np.random.shuffle(points)
        pt1, pts1_rem, pt2, pts2_rem = points[:4, :2],  points[4:, :2], points[:4, 2:], points[4:, 2:]

        H = homography(pt1, pt2)

        inliers = [(pts1_rem[i], pts2_rem[i]) for i in range(len(pts1_rem)) if d(pts1_rem[i], pts2_rem[i], H) < 100]

        if len(inliers) > inlier_count:
            inlier_count = len(inliers)
            selected_inliers = np.array(inliers)
    
    H = homography(selected_inliers[:, 0], selected_inliers[:, 1])
    return H

H = RANSAC(pt1, pt2)
print(H)

file = open(r"graf\H1to5p", "r")
H = []

for i in range(3):
    H.append(tuple(map(float, file.readline().strip().split())))
H = np.array(H)
img_p = cv.warpPerspective(img1, H, (img5.shape[1], img5.shape[0]))
ret, threshold = cv.threshold(img_p, 10, 1, cv.THRESH_BINARY_INV)
img2_thresholded = np.multiply(threshold, img5)
img_blended = cv.addWeighted(img2_thresholded, 1, img_p, 1, 0)

fig, ax = plt.subplots(1, 2, figsize=(18, 20))
ax[0].imshow(img5)
ax[0].set_title("img5.ppm")
ax[0].axis("off")
ax[1].imshow(img_blended)
ax[1].set_title("Stitched Image")
ax[1].axis("off")

plt.show()

