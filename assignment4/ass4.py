# %% [markdown]
# # Assignment 4: Feature points, matching, homography

# %% [markdown]
# ### Exercise 1: Feature points detectors

# %% [markdown]
# #### (a) Hessian detector

# %% [markdown]
# Q: What kind of structures are detected by the algorithm? How does the parameter sigma affect the result? \
# A: Corners. Changes the size of corners detected.

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# %%
def normalize(xs):
    res = [abs(x) for x in xs]
    return xs / np.sum(res)
    
def gaussdx(sigma):
    return np.array(normalize([(-1/(np.sqrt(2 * np.pi) * sigma**3)) * x * np.exp(-x**2 / (2 * sigma**2)) for x in np.arange(-np.ceil(3*sigma), np.ceil(3*sigma)+1, 1)]))

# %%
def gaussian_kernel(sigma):
    return np.array([1 / (np.sqrt(2*np.pi) * sigma) * np.exp((-np.square(x)) / (2 * np.square(sigma))) for x in np.arange(-np.ceil(3*sigma), np.ceil(3*sigma)+1, 1)])

# %%
def derive_1(img, sigma):
    img = img.astype(np.float64)

    gx = gaussian_kernel(sigma)
    gy = np.reshape(gx, (1, -1))
    
    d_gx = gaussdx(sigma)
    d_gy = np.reshape(d_gx, (1, -1))
    
    # switch x and y because filter2D will use (256, ) shape as (1, 256)
    gx, gy, d_gx, d_gy = gy, gx, d_gy, d_gx
    
    dx_img = cv2.filter2D(cv2.filter2D(img, -1, gy), -1, d_gx)
    dy_img = cv2.filter2D(cv2.filter2D(img, -1, gx), -1, d_gy)
    
    return dx_img, dy_img

def derive_2(img, sigma):
    #img = img.astype(np.float64)
    gx = gaussian_kernel(sigma)
    gy = np.reshape(gx, (1, -1))
    
    d_gx = gaussdx(sigma)
    d_gy = np.reshape(d_gx, (1, -1))
    
    dx_img, dy_img = derive_1(img, sigma)
    
    dx_img = cv2.filter2D(dx_img, -1, gx)
    dy_img = cv2.filter2D(dy_img, -1, gy)
    
    # switch x and y because filter2D will use (256, ) shape as (1, 256)
    dxx_img = cv2.filter2D(cv2.filter2D(dx_img, -1, gx), -1, d_gy)
    dxy_img = cv2.filter2D(cv2.filter2D(dx_img, -1, gy), -1, d_gx)
    dyy_img = cv2.filter2D(cv2.filter2D(dy_img, -1, gx), -1, d_gy)
    
    return dxx_img, dxy_img, dyy_img


# %%
def nonmaxima_suppression(img, boxsize, thresh=0.004):
    res = img.copy()
    for i in range(1,img.shape[0]-boxsize):
        for j in range(1,img.shape[1]-boxsize):                
            neighborhood = [img[i-boxsize,j-boxsize], img[i-boxsize,j], img[i-boxsize, j+boxsize],
                    img[i,j-boxsize], img[i,j+boxsize], 
                    img[i+boxsize,j-boxsize], img[i+boxsize,j], img[i+boxsize,j+boxsize]]
            
            # Check if its the strongest
            if np.max(neighborhood) > res[i,j]:
                # Remove multiple maximums
                if np.max(neighborhood) == img[i,j]:
                    np.where(neighborhood == res[i,j], 0, res[i,j])
                
                res[i,j] = 0
    return res
            

# %%
def hessian_points(img, sigma=3, thresh=0.004):
    '''
        Takes in grayscale image, return Hessian determinant
    '''
    Ixx, Ixy, Iyy = derive_2(img, sigma)
    
    det = Ixx * Iyy - np.square(Ixy)
    det[det < thresh] = 0
    return det   

# %%
def plot_points(img, thresh):
    '''
        Takes in image determinant and plots all points on the original image
    '''
    xs = []
    ys = []
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
           if img[y,x] > thresh:
               xs.append(x)
               ys.append(y)
    return xs, ys 

# %%
img_test_orig = cv2.imread("data/graf/graf_a.jpg")
img_test_orig = cv2.cvtColor(img_test_orig, cv2.COLOR_BGR2GRAY)

n_neighbors = 3

# Sigma = 3
det_test_3 = hessian_points(img_test_orig, 3)
det_test_3 = nonmaxima_suppression(det_test_3, n_neighbors)

# Sigma = 6
det_test_6 = hessian_points(img_test_orig, 6)
det_test_6 = nonmaxima_suppression(det_test_6, n_neighbors)

# Sigma = 9
det_test_9 = hessian_points(img_test_orig, 9)
det_test_9 = nonmaxima_suppression(det_test_9, n_neighbors)

# %%
fig, ax = plt.subplots(2,3,figsize=(10,6))

xs3, ys3 = plot_points(det_test_3, 50)
ax[0,0].imshow(det_test_3, cmap="gray")
ax[0,0].set_title("sigma = 3")

ax[1,0].scatter(xs3, ys3, marker="x", c="red", s=5)
ax[1,0].imshow(img_test_orig, cmap="gray")

xs6, ys6 = plot_points(det_test_6, 50)
ax[0,1].imshow(det_test_6, cmap="gray")
ax[0,1].set_title("sigma = 6")

ax[1,1].scatter(xs6, ys6, marker="x", c="red", s=5)
ax[1,1].imshow(img_test_orig, cmap="gray")

xs9, ys9 = plot_points(det_test_9, 50)
ax[0,2].imshow(det_test_9, cmap="gray" )
ax[0,2].set_title("sigma = 9")

ax[1,2].scatter(xs9, ys9, marker="x", c="red", s=5)
ax[1,2].imshow(img_test_orig, cmap="gray")

# %% [markdown]
# #### (b) Harris detector

# %%
smoothing_sigma_factor = 1.6

# %%
def C_mat(img, sigma):
    i = img.copy()
    Ix, Iy = derive_1(i, sigma * smoothing_sigma_factor)
        
    # smoothe img with gauss
    kernel = gaussian_kernel(sigma * smoothing_sigma_factor)
    kernel_T = np.reshape(kernel, (1, -1))
    i = cv2.filter2D(i, -1, kernel)
    
    # Get values
    sxx = cv2.filter2D(cv2.filter2D(np.square(Ix), -1, kernel), -1, kernel_T)
    sxy = cv2.filter2D(cv2.filter2D((Ix * Iy), -1, kernel), -1, kernel_T);
    syy = cv2.filter2D(cv2.filter2D(np.square(Iy), -1, kernel.T), -1, kernel_T)
    
    
    return np.array([[sxx, sxy], [sxy, syy]]) 

# %%
def check_corner(img_orig, C, thresh):
    img = img_orig.copy()
    alpha = 0.06
    
    det = (C[0,0] * C[1,1]) - np.square(C[1,0])
    trace = C[0,0] + C[1,1]
        
    #img[det - alpha * np.square(trace) < thresh] = 0        

    return det - alpha * np.square(trace) #img

# %%
n_neighbors = 6
thresh = 1e-7

C3 = C_mat(img_test_orig, 3)
corner_map3 = check_corner(img_test_orig, C3, thresh)
corner_map3 = nonmaxima_suppression(corner_map3, n_neighbors)
#corner_map3 /= np.max(corner_map3)

C6 = C_mat(img_test_orig, 6)
corner_map6 = check_corner(img_test_orig, C6, thresh)
corner_map6 = nonmaxima_suppression(corner_map6, n_neighbors)
#corner_map6 /= np.max(corner_map6)

C9 = C_mat(img_test_orig, 9)
corner_map9 = check_corner(img_test_orig, C9, thresh)
corner_map9 = nonmaxima_suppression(corner_map9, n_neighbors)
#corner_map9 /= np.max(corner_map9)

# %%
thresh = 70

fig, ax = plt.subplots(2, 3, figsize=(12,6))

xs3, ys3 = plot_points(corner_map3, thresh)
ax[0,0].imshow(corner_map3, cmap="gray")
ax[0,0].set_title("sigma = 3")

ax[1,0].scatter(xs3, ys3, marker="x", c="r", s=.1)
ax[1,0].imshow(img_test_orig, cmap="gray")


xs6, ys6 = plot_points(corner_map6, thresh)
ax[0,1].imshow(corner_map6, cmap="gray")
ax[0,1].set_title("sigma = 6")

ax[1,1].scatter(xs6, ys6, marker="x", c="r", s=.1)
ax[1,1].imshow(img_test_orig, cmap="gray")


xs9, ys9 = plot_points(corner_map9, thresh)
ax[0,2].imshow(corner_map9, cmap="gray")
ax[0,2].set_title("sigma = 9")

ax[1,2].scatter(xs9, ys9, marker="x", c="r", s=.1)
ax[1,2].imshow(img_test_orig, cmap="gray")

# %% [markdown]
# ### Exercise 2: Matching local regions

# %% [markdown]
# #### (a) Finding correspondences

# %%
from a4_utils import *

# %%
def hellinger(h1, h2):
    return np.sqrt(0.5 * np.sum(np.square(np.sqrt(h1) - np.sqrt(h2))))**2

# %%
def find_correspondences(descriptor1, descriptor2):
    '''
        Takes in 2 arrays of feature point descriptors and matches each one from the first array to the closest one from the second one
        Returns an array of pairs of indices
    '''
    indices = []
    for i in range(len(descriptor1)):
        d1 = descriptor1[i]
        
        min_dist = 10000;
        min_desc = -1
              
        for j in range(len(descriptor2)):
            d2 = descriptor2[j]
            
            dist = hellinger(d1, d2)
            if dist < min_dist:
                min_dist = dist
                min_desc = j
        indices.append((i, min_desc))
    return indices

# %%
# Read in the images
img_a = cv2.imread("data/graf/graf_a_small.jpg")
img_b = cv2.imread("data/graf/graf_b_small.jpg")

img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

determinant_thresh = 25
sigma = 3
n_neighbors = 1

# Get feature points
det_a = hessian_points(img_a, sigma)
det_a = nonmaxima_suppression(det_a, n_neighbors)
a_xs, a_ys = plot_points(det_a, determinant_thresh)

det_b = hessian_points(img_b, sigma)
det_b = nonmaxima_suppression(det_b, n_neighbors)
b_xs, b_ys = plot_points(det_b, determinant_thresh)

img_a = img_a.astype(np.float64) / 255
img_b = img_b.astype(np.float64) / 255

# Get descriptors for found feature points
desc1 = simple_descriptors(img_a, a_ys, a_xs)
desc2 = simple_descriptors(img_b, b_ys, b_xs)

# Match closest descriptors
indices = find_correspondences(desc1, desc2)

# Get x and y coordinates
feature_points_a, feature_points_b = [], []

for index in indices:
    feature_points_a.append([a_xs[index[0]], a_ys[index[0]]])
    feature_points_b.append([b_xs[index[1]], b_ys[index[1]]])

# Display results
display_matches(img_a, feature_points_a, img_b, feature_points_b)

# %% [markdown]
# #### (b) Feature point matching

# %%
def harris(img, sigma, thresh):
    C = C_mat(img, sigma)
    corner_map = check_corner(img, C, thresh)
    corner_map = nonmaxima_suppression(corner_map, 8)
    corner_map /= np.max(corner_map)
    xs, ys = plot_points(corner_map, .05)
    
    return np.array([xs, ys])

# %%
def hessian(img, sigma, thresh):
    # Get feature points
    det_a = hessian_points(img, thresh, sigma)
    det_a = nonmaxima_suppression(det_a, 6)
    xs, ys = plot_points(det_a, 22)
    
    return np.array([xs, ys])

# %%
def get_symmetrical_correspondences(i1, i2):
    '''
        Returns an array of only the symmetrical matches of indices
        if i1[x] = y then i2[y] = x 
    '''
    
    correspondences = []
    
    for pair in i1:
        reverse = pair[::-1]
        if reverse in i2:
            correspondences.append(pair)     
                
    return correspondences   

# %%
def find_matches(img1, img2, sigma, detector="hessian", draw=True, thresh=0.1):
    '''
        Takes in 2 images and returns array of index pairs
    '''

    # Get feature points
    if detector == "harris":
        fpoints1 = harris(img1, sigma, 1e-5)
        fpoints2 = harris(img2, sigma, 1e-5)
    
    # Get feature points
    if detector == "hessian":
        fpoints1 = hessian(img1, sigma, .9)
        fpoints2 = hessian(img2, sigma, .9)
    
    print("Got fpoints", fpoints1.shape)
    
    img1 = img1.astype(np.float64) / 255
    img2 = img2.astype(np.float64) / 255
    
    # Get simple descriptors
    desc1 = simple_descriptors(img1, fpoints1[1], fpoints1[0])
    desc2 = simple_descriptors(img2, fpoints2[1], fpoints2[0])
    print("Got descriptors")
    
    # Find corresponding descriptors
    indices1 = find_correspondences(desc1, desc2)
    indices2 = find_correspondences(desc2, desc1)
    
    indices = get_symmetrical_correspondences(indices1, indices2)
    print("Got indices")    

    # Get x and y coordinates
    feature_points_a, feature_points_b = [], []

    for index in indices:
        feature_points_a.append([fpoints1[0][index[0]], fpoints1[1][index[0]]])
        feature_points_b.append([fpoints2[0][index[1]], fpoints2[1][index[1]]])
    
    if draw:
        display_matches(img1, feature_points_a, img2, feature_points_b)
    return np.array(feature_points_a), np.array(feature_points_b)

# %%
img_a = cv2.imread("data/graf/graf_a_small.jpg")
img_b = cv2.imread("data/graf/graf_b_small.jpg")

img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

fp1, fp2 = find_matches(img_a, img_b, 3, "hessian")

# %% [markdown]
# Q: What do you notice when visualizing the correspondences? How accurate are the matches? \
# A: A lot of areas have lots of corresponding points close together (not eliminated by non maxima suppression), so they are still clustered together, whereas some of the weak correspondences are lost, because they might not be symmetrical or not strong enough. The matches that do remain are relatively accurate.

# %% [markdown]
# #### (c) (25pts) Implement SIFT

# %%
def gaussian_filter_2d(img, sigma):
    k = gaussian_kernel(sigma)
    k_T = np.reshape(k, (1, -1))
    img = cv2.filter2D(cv2.filter2D(img, -1, k), -1, k_T)
  
    return img

# %%
def upscale_img(img, sigma):
    # resize
    img = cv2.resize(img, (0,0), fx=2, fy=2, interpolation=1)
    
    # blur
    img = gaussian_filter_2d(img, sigma)    
           
    return img

# %%
def downscale_img(img):
    '''
        Halves the image
    '''
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=1)

# %%
def pyramid_height(img_shape):
    '''
        How many times can we downscale the image 
    '''
    return int(np.round(np.log(min(img_shape)) / np.log(2) - 1))

# %%
def get_gauss_kernel_sigmas(sigma, n_intervals=3):
    '''
        Generates an array of sigmas for each image in the pyramid
    '''
    IMG_PER_LAYER = n_intervals + 3
    
    k = 2 ** (1 / n_intervals)
    kernel_sigmas = np.zeros(IMG_PER_LAYER)
    kernel_sigmas[0] = sigma
    
    for i in range(1, IMG_PER_LAYER):
        prev = (k ** (i-1)) * sigma
        current = k * prev
        kernel_sigmas[i] = np.sqrt(np.square(current) - np.square(prev))
    return np.array(kernel_sigmas)  
       

# %%
def get_pyramid_images(img, pyramid_height, kernel_sigmas):
    '''
        Get pyramid of gaussian images
    '''
    imgs = []
    
    for i in range(pyramid_height):
        imgs_in_layer = [img]
        
        # Blur the other ones
        for sigma in kernel_sigmas:
            img = gaussian_filter_2d(img, sigma)
            imgs_in_layer.append(img)
        
        imgs.append(imgs_in_layer)
        base = imgs_in_layer[-3]    # Has the base blur
        img = downscale_img(base)

    return np.array(imgs, dtype='object')        


# %%
        
def get_DoGs(images):
    '''
        Applies DoG to image pyramid
    '''
    
    dogs = []
    
    # Do it for each layer
    for layer in images:
        dogs_in_layer = []
        # Get all pairs
        for img1, img2 in zip(layer, layer[1:]):
            dog = np.subtract(img2, img1)
            dogs_in_layer.append(dog)
        dogs.append(dogs_in_layer)
    return np.array(dogs, dtype='object')
    
        
    

# %%
def get_DoG(img, sigma):
    # generate base image
    img = upscale_img(img, sigma)
    
    # compute height
    height = pyramid_height(img.shape)
    
    # generate gaussian kernels
    kernel_sigmas = get_gauss_kernel_sigmas(sigma, 3)
    
    # generate gaussian images
    img_gauss = get_pyramid_images(img, height, kernel_sigmas)
    
    # generateDoGImages
    img_dog = get_DoGs(img_gauss)
    
    return img_gauss, img_dog

# %%
def checkNeighborhood(img1, img2, img3, threshold):
    """
        Check if the center element of the 3x3x3 array is greater than or less than all its neighbors
    """
    try:
        center_pixel_value = img2[1, 1]
    except IndexError:
        return False
    
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return np.max(img1) == center_pixel_value and \
                   np.max(img3) == center_pixel_value and \
                   np.max(img2[0, :]) == center_pixel_value and \
                   np.max(img2[1, :]) == center_pixel_value and \
                   center_pixel_value >= img2[1, 0]
        elif center_pixel_value < 0:
            return np.min(img1) == center_pixel_value and \
                   np.min(img3) == center_pixel_value and \
                   np.min(img2[0, :]) == center_pixel_value and \
                   np.min(img2[1, :]) == center_pixel_value and \
                   center_pixel_value <= img2[1, 0]
    return False

# %%
def get_extremes(img_dog, n_intervals=3):
    '''
        Finds positions of pixels of extremes in the image pyramid
    '''
    threshold = np.floor(0.5 * 0.04 / n_intervals * 255)  # OpenCV implementation
    fpoints = []
    
    for ix, layer in enumerate(img_dog):
        for img_ix, (img1, img2, img3) in enumerate(zip(layer, layer[1:], layer[2:])):
            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    # Check 8 neighborhood
                    if checkNeighborhood(img1[i-1:i+2, j-1:j+2], img2[i-1:i+2, j-1:j+2], img3[i-1:i+2, j-1:j+2], threshold):
                        fpoints.append((i,j))
                        
    return fpoints

# %%
def sift(img, sigma):
    '''
        Accepts float image, sigma, returns keypoints and descriptors
    '''
    
    img_gauss, img_dog = get_DoG(img, sigma)
    
    fpoints = get_extremes(img_dog, sigma)
    
    return img_dog, fpoints  
    

# %%
img = cv2.imread("data/graf/graf_a_small.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
img_dog, fpoints = sift(img, 1.6)

# Plot all the
#fig, ax = plt.subplots(img_dog.shape[0], img_dog.shape[1], figsize=(20,18))
fig, ax = plt.subplots(3,3, figsize=(20,18))

ix = 0
for layer in img_dog:
    if ix >= 3: break
    jx = 0
    for img in layer:
        if jx >= 3:
            break
        ax[ix, jx].imshow(img, cmap="gray")
        ax[ix, jx].set_title(f"img {ix + jx}")
        jx += 1
    ix += 1
#plt.imshow(img_dog[3,2], cmap="gray")


# %%
img = cv2.imread("data/graf/graf_a_small.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

xs = filter(lambda x: x[0], fpoints)
ys = filter(lambda x: x[1], fpoints)

for x,y in zip(xs, ys):
    plt.scatter(x, y, marker="x", c="r")
    
plt.imshow(img, cmap="gray")

# %% [markdown]
# ### Exercise 3: Homography estimation

# %% [markdown]
# Q: Looking at the equation above, which parameters account for translation and which for rotation and scale? \
# A: p1 is parameter for rotation, p2 for scaling and p3 and p4 are translation parameters

# %% [markdown]
# Q: Write down a sketch of an algorithm to determine similarity transform from a set of point correspondences \
# `P = [(x r1 , x t1 ), (x r2 , x t2 ), . . . (x rn , x tn )]`. For more details consult the lecture notes.

# %%
def get_Ai(src_point, dest_point):
    x_r, y_r = src_point
    x_t, y_t = dest_point
    z_r, z_t = 1, 1         # Homogenous coord
    
    # 2x9 Matrix
    A_partial = np.array([
        [x_r, y_r, 1, 0, 0, 0, -x_t * x_r, -x_t * y_r, -x_t],
        [0, 0, 0, x_r, y_r, 1, -y_t * x_r, -y_t * y_r, -y_t]
    ])
    return A_partial

# %%
def get_A(points1, points2):
    '''
        Piles Ai matrices on top of each other to gain matrix A
    '''
    A = []
    
    for i in range(points1.shape[0]):
        Ai = get_Ai(points1[i], points2[i])
        A.append(Ai)
        
    # We need 2D matrix not 3D
    return np.concatenate(A, axis=0)

    

# %%
def estimate_homography(fpoints):
    # Construct A by Ah = 0
    A = get_A(fpoints[0], fpoints[1])
    
    # SVD -> U, S, VT = np.linalg.svd(A)
    U, S, VT = np.linalg.svd(A)

    # Compute h
    h = VT[-1].reshape((3,3))
    h /= h[2,2]
    
    return np.array(h) 

# %%
img1 = cv2.imread("data/newyork/newyork_a.jpg")
img2 = cv2.imread("data/newyork/newyork_b.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2  = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

points = np.loadtxt("data/newyork/newyork.txt")

# x1, y1
fpoints1 = []
# x2, y2
fpoints2 = []

for point in points:
    fpoints1.append([point[0], point[1]])
    fpoints2.append([point[2], point[3]])

fpoints = np.array([fpoints1, fpoints2])
H = estimate_homography(fpoints)
hardcoded_H = H
print(H)
display_matches(img1, fpoints1, img2, fpoints2)

# %%
img = cv2.warpPerspective(img1, H, (300,300))
fig, ax = plt.subplots(1, 3, figsize=(10, 7))
ax[0].imshow(img1, cmap="gray")
ax[1].imshow(img2, cmap="gray")
ax[2].imshow(img, cmap="gray")

# %% [markdown]
# ####  (b) Detection and RANSAC

# %%
def euclidean(h1, h2):
    return np.sqrt(np.sum(np.square(h1 - h2)))

# %%
def get_reprojection_error(fpoints: np.ndarray, H: np.ndarray):
    '''
        Takes in a pair of points and returns average cost
    '''
    
    fpoints1, fpoints2 = fpoints                
    x, y = fpoints1
    z = 1
    transformed = np.matmul(H, [x,y,z])   
    
    fpoints2 = [fpoints2[0], fpoints2[1], 1]      
   
    price = euclidean(transformed, fpoints2)
    return price

# %%
def get_inliers(fpoints1, fpoints2, H, thresh=100):
    inliers = []
    min_cost = 1000000
    
    for i in range(len(fpoints1)):
        cost = get_reprojection_error([fpoints1[i], fpoints2[i]], H)
        if cost < min_cost:
            min_cost = cost
        if cost < thresh:
            inliers.append([fpoints1[i], fpoints2[i]])
            
    return inliers, min_cost

# %%
def ransac(fpoints1, fpoints2, thresh: float or int, n_iter: int = 10):
    #fpoints1, fpoints2 = find_matches(img1, img2, 3, "harris", False)
    max_inliers = []
    best_H = None
    min_cost = 10000
    use_inliers = False
            
    for i in range(n_iter):

        # Get 4 random points
        if not use_inliers:
            indices = np.random.randint(fpoints1.shape[0], size=4)
            fpoints_to_estimate = np.array([fpoints1[indices], fpoints2[indices]]) 
        use_inliers = False
        # Estimate homography
        H = estimate_homography(fpoints_to_estimate)
                        
        # Get inliers
        inliers, cost = get_inliers(fpoints1, fpoints2, H, thresh)
                
        if np.array(inliers).shape[0] and np.array(inliers).shape[1] >= 4:
            fpoints = inliers
            use_inliers = True
                
        if cost < min_cost:
            min_cost = cost
        # Store best results
        if len(inliers) > len(max_inliers) or len(max_inliers) == 0:
            max_inliers = inliers
            best_H = H
            
        print(f"n_inliers: {len(inliers)}, max inliers: {len(max_inliers)}, min_cost: {min_cost}")
        
            
    return best_H, max_inliers    
            

# %%
img1 = cv2.imread("data/newyork/newyork_a.jpg")
img2 = cv2.imread("data/newyork/newyork_b.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2  = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

fpoints1, fpoints2 = find_matches(img1, img2, 3, "hessian", 1)

# %%
img1 = cv2.imread("data/newyork/newyork_a.jpg")
img2 = cv2.imread("data/newyork/newyork_b.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2  = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

H, inliers = ransac(fpoints1, fpoints2, 50, 10)

# %%
img1 = img1.astype(np.float64)
print(H)
warped = cv2.warpPerspective(img1, H, (img2.shape[0], img2.shape[1]))

#plt.imshow(img2, cmap="gray")
#plt.imshow(warped, alpha=.5, cmap="gray")

fig, ax = plt.subplots(1,2,figsize=(10,6))

ax[0].imshow(img2, cmap="gray")
ax[0].set_title("newyork_b")

ax[1].imshow(warped, cmap="gray")
ax[1].set_title("transformed_newyork_a")


# %%
diff = np.abs(hardcoded_H - H)
print(diff)

# %% [markdown]
#  #### (e)  Custom warpPerspective (10pts)

# %%
def check_neighbors(transformed: np.ndarray, x: float, y: float):
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y))
    
    pairs = [
        [x1, y1],
        [x1, y2],
        [x2, y1],
        [x2, y2]
    ]
    
    for pair in pairs:
        x, y = pair
        print(x, y)
        if x >= transformed.shape[0] or x < 0 or y >= transformed.shape[1] or y < 0:
            continue 
        if transformed[x, y] == 0:
            return x, y

# %%
def warp_persp(img, H):
    '''
        Works like cv2.warpPerspective, uses image width and height
    '''
    
    transformed = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x,y,z = i,j,1
            
            warped_point = np.matmul(H, [x,y,z])
            
            #if warped_point[2] != 1:
            #    warped_point /= warped_point[1]
                
            x,y,z = warped_point
            
            if x >= img.shape[0] or x < 0 or y >= img.shape[1] or y < 0:
                continue 
            
            try:
                x, y = check_neighbors(transformed, x, y)
            except BaseException:
                x, y = int(np.round(x)), int(np.round(y))
                
                if x >= transformed.shape[0] or x < 0 or y >= transformed.shape[1] or y < 0:
                    continue;
            
            
            
            transformed[x,y] = img[i,j]
            #transformed /= transformed[2]
    return transformed   
                       
    

# %%
img1 = cv2.imread("data/newyork/newyork_a.jpg")
img2 = cv2.imread("data/newyork/newyork_b.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2  = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

H = H
warped = warp_persp(img1, H)

fig,ax = plt.subplots(1,3, figsize=(15,5))

ax[0].imshow(img1, cmap="gray")
ax[1].imshow(img2, cmap="gray")
ax[2].imshow(warped, cmap="gray")


