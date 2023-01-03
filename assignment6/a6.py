# %% [markdown]
# # Assignment 6: Reduction of dimensionality and recognition

# %% [markdown]
# ### Exercise 1: Direct PCA method

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# %% [markdown]
# #### a) Analytically compute eigenvectors and eigenvalues given four points.

# %%
plt.imshow(cv2.imread("1a.jpg"))

# %% [markdown]
# #### b) Calculate and visualize PCA for 2D data from `points.txt`.

# %%
from a6_utils import *

# %%
def get_covariance_matrix(data: np.ndarray) -> np.ndarray:
    '''
        Calculates the covariance matrix
    '''
    
    data = data.T
    
    # Calculate the covariance matrix
    covariance_matrix = data.T @ data / (data.shape[0] -1)
    
    return covariance_matrix

# %%
def pca(pts: np.ndarray) -> np.ndarray:

    mean = np.mean(pts, axis=0)
    centered = pts - mean
            
    cov = get_covariance_matrix(centered.T)
    
    U,S,VT = np.linalg.svd(cov)
    #S = S[:]
    #U = U[:, :2]

    #projected = np.matmul(U, centered)
    #reconstructed = np.matmul(U, projected) + mean
    
    return cov, (U, S)

# %%

orig_points = np.loadtxt("data/points.txt")
points = orig_points.copy()
cov, (eigenvectors, eigenvalues) = pca(points)
mu = np.mean(points, axis=0)
drawEllipse(mu, cov)

plt.scatter(list(map(lambda x: x[0], points)), list(map(lambda x: x[1], points)))

for pt in range(len(points)):
    plt.annotate(pt+1, (points[pt][0], points[pt][1]))
   
# C
plt.plot([mu[0], mu[0] + eigenvectors[0,0] * np.sqrt(eigenvalues[0])], [mu[1], mu[1] + eigenvectors[0,1] * np.sqrt(eigenvalues[0])], color="red")
plt.plot([mu[0], mu[0] + eigenvectors[1,0] * np.sqrt(eigenvalues[1])], [mu[1], mu[1] + eigenvectors[1,1] * np.sqrt(eigenvalues[1])], color="green")


# %% [markdown]
# Q: What do you notice about the relationship between the eigenvectors and the data? What happens to the eigenvectors if you change the data or add more
# points?
# A: Their orientation changes. The eigenvectors show the direction of the biggest variance, so additional points may change them.

# %% [markdown]
# #### d) Plot the cumulative graph of eighenvalues and normalize it

# %%
print(eigenvalues)
cumulative = np.cumsum(eigenvalues)
cumulative = cumulative / np.max(cumulative)
plt.bar([i for i in range(len(eigenvalues))], cumulative)

# %% [markdown]
# The first eigenvector alone explains 80% of the variance

# %% [markdown]
# #### e) Project data into the subspace of the first eigenvector

# %%
eigenvalues[1] = 0

centered_points = points - mu
vector = eigenvectors[0]

projected_points = centered_points @ eigenvectors[0]
projected_points = np.reshape(projected_points, (5,1))
vector = np.reshape(vector, (1,2))

reconstructed_points = projected_points @ vector + mu

plt.scatter(reconstructed_points[:,0], reconstructed_points[:,1])
for pt in range(len(points)):
    plt.annotate(pt+1, (reconstructed_points[pt][0], reconstructed_points[pt][1]))
   
# C
plt.plot([mu[0], mu[0] + eigenvectors[0,0] * np.sqrt(eigenvalues[0])], [mu[1], mu[1] + eigenvectors[0,1] * np.sqrt(eigenvalues[0])], color="red")


# %% [markdown]
# The data is projected onto the eigenvector. They are lying on a line.

# %% [markdown]
# #### f) Use q_point = [6, 6] and get the closest point. Project all the points to PCA subspace, and check which point is closest then.

# %%
q_point = [6,6]

# Get closest point
min_dist = np.inf
closest_point = 0
ix = 1
for p in points:
    dist = np.sqrt(np.square(p[0] - 6) + np.square(p[1] - 6))
    if dist < min_dist:
        min_dist = dist
        closest_point = ix
    ix += 1

print("Closest point:",closest_point)


projected_q = q_point @ eigenvectors[0]
projected_q = np.reshape(projected_q, (1,1))
reconstructed_q = projected_q @ vector + mu 
reconstructed_q = np.reshape(reconstructed_q, (2,))
print(reconstructed_q)

plt.scatter(reconstructed_points[:,0], reconstructed_points[:,1])
for pt in range(len(points)):
    plt.annotate(pt+1, (reconstructed_points[pt][0], reconstructed_points[pt][1]))
plt.scatter(reconstructed_q[0], reconstructed_q[1])
plt.annotate("q_point", (reconstructed_q[0], reconstructed_q[1]))
plt.plot([mu[0], mu[0] + eigenvectors[0,0] * np.sqrt(eigenvalues[0])], [mu[1], mu[1] + eigenvectors[0,1] * np.sqrt(eigenvalues[0])], color="red")

min_dist = np.inf
closest_point = 0
ix = 1
for p in reconstructed_points:
    dist = np.sqrt(np.square(p[0] - reconstructed_q[0]) + np.square(p[1] - reconstructed_q[1]))
    if dist < min_dist:
        min_dist = dist
        closest_point = ix
    ix += 1    
print("New closest point:", closest_point) 
    


# %% [markdown]
# ### Exercise 2: The dual PCA

# %% [markdown]
# #### a) Implement the dual method and test it using the data from `points.txt`

# %%
def get_covariance_matrix(data: np.ndarray) -> np.ndarray:
    '''
        Calculates the covariance matrix
    '''
    
    data = data.T
    
    # Calculate the covariance matrix
    covariance_matrix = data.T @ data / (data.shape[0] -1)
    
    return covariance_matrix

# %%
def dual_pca(pts: np.ndarray) -> np.ndarray:

    mean = np.mean(pts, axis=0)
    centered = pts - mean
            
    cov = get_covariance_matrix(centered.T)
    
    U,S,VT = np.linalg.svd(cov)
    for i in range(len(S)):
        S[i] += 1e-15
    
    U = pts @ U * (np.sqrt(1 / (S * (points.shape[0] - 1))))
    
    return cov, (U, S)

# %% [markdown]
# #### b) Project data from exercise 1

# %%
cov, (eigenvectors, eigenvalues) = dual_pca(points)

centered_points = points - mu

projected_points = eigenvectors.T @ (centered_points)
reconstructed_points = eigenvectors @ projected_points 

# Original points
plt.scatter(points[:, 0], points[:, 1], color="orange")

# Reconstructed points
plt.scatter(reconstructed_points[:,0], reconstructed_points[:,1])
for pt in range(len(points)):
    plt.annotate(pt+1, (reconstructed_points[pt][0], reconstructed_points[pt][1]))
   
# C
plt.plot([mu[0], mu[0] + eigenvectors[0,0] * np.sqrt(eigenvalues[0])], [mu[1], mu[1] + eigenvectors[0,1] * np.sqrt(eigenvalues[0])], color="red")


# %% [markdown]
# ### Exercise 3: Image decomposition examples

# %% [markdown]
# #### a) Data preparation

# %%
import os, pathlib
def load_images(path: str) -> np.ndarray:
    
    matrix = []
    for file in os.listdir(path):
        img = cv2.imread(f"{path}/{file}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        column = np.reshape(img, (1, -1)) 
        matrix.append(column.T)
    
    matrix = np.array(matrix)
    matrix = np.reshape(matrix, (64, -1))
    matrix = matrix.T
    return matrix

# %%
matrix = load_images("data/faces/1")
print(matrix.shape)

# %% [markdown]
# #### b) Using dual PCA

# %%
def normalize(img):
    return ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255

# %%
def get_eigenvectors(image_matrix: np.ndarray) -> np.ndarray:
    cov, (evec, evals) = dual_pca(image_matrix)
    return evec

# %%
vectors = get_eigenvectors(matrix)
vectors = vectors.T

fig, ax = plt.subplots(1,5,figsize=(10,10))
first_five = []
for i in range(5):
    vec = np.reshape(vectors[i], (96, 84))
    ax[i].imshow(vec, cmap="gray")
    first_five.append(vec)
    

# %%
img = cv2.imread("data/faces/1/001.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.reshape(img, (-1, 1))

cov, (evec, evals) = dual_pca(matrix)

mu = np.mean(img)

corrupted = img.copy()
corrupted[4074] = 0

projected_points = evec.T @ (img - mu)
projected_corrupted = evec.T @ (corrupted - mu)

corrupted_pca = projected_points.copy()
corrupted_pca[4] = 0
reconstructed_points = evec @ projected_points + mu
reconstructed_corrupted = evec @ projected_corrupted + mu
reconstructed_corrupted_pca = evec @ corrupted_pca + mu

fig, ax = plt.subplots(2,4, figsize=(12, 8))

ax[0,0].imshow(img.reshape(96,84), cmap="gray")
ax[0,1].imshow(corrupted.reshape(96,84), cmap="gray")
ax[0,2].imshow(reconstructed_corrupted.reshape(96,84), cmap="gray")
ax[0,3].imshow((normalize(img) - normalize(reconstructed_corrupted)).reshape(96,84), )

ax[0,0].set_title("original")
ax[0,1].set_title("corrupted")
ax[0,2].set_title("reconstructed")
ax[0,3].set_title("1-3 diff")

ax[1,0].imshow(reconstructed_points.reshape(96,84), cmap="gray")
ax[1,0].set_title("orig - pca - rec")
ax[1,1].imshow(reconstructed_corrupted.reshape(96,84), cmap="gray")
ax[1,1].set_title("corrupted - pca - rec")
ax[1,2].imshow(reconstructed_corrupted_pca.reshape(96,84), cmap="gray")
ax[1,2].set_title("orig - corrupted - rec")
ax[1,3].axis("off")

# %% [markdown]
# Changing a pixel in image space has a great effect on the reconstructed points, but getting rid of an eigenvector gets rid of one of the principal components - we lose a significant part of important features, depending on which eigenvector we set to zero

# %% [markdown]
# #### c) Effect of the number of components

# %%
img = cv2.imread("data/faces/1/001.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.reshape((-1, 1))
img = img.astype(np.float64)

cov, (evec, evals) = dual_pca(matrix)
projected = evec.T @ (img - mu)

p1 = projected.copy()
p1[1:] = 0
r1 = evec @ p1 + mu

p2 = projected.copy()
p2[2:] = 0
r2 = evec @ p2 + mu

p4 = projected.copy()
p4[4:] = 0
r4 = evec @ p4 + mu

p8 = projected.copy()
p8[8:] = 0
r8 = evec @ p8 + mu

p16 = projected.copy()
p16[16:] = 0
r16 = evec @ p16 + mu

p32 = projected.copy()
p32[32:] = 0
r32 = evec @ p32 + mu

fig, ax = plt.subplots(1,7,figsize=(18,12))


ax[0].imshow(img.reshape(96,84), cmap="gray")

ax[1].imshow(r32.reshape(96,84), cmap="gray")
ax[1].set_title("32c")

ax[2].imshow(r16.reshape(96,84), cmap="gray")
ax[2].set_title("16c")

ax[3].imshow(r8.reshape(96,84), cmap="gray")
ax[3].set_title("8c")

ax[4].imshow(r4.reshape(96,84), cmap="gray")
ax[4].set_title("4c")

ax[5].imshow(r2.reshape(96,84), cmap="gray")
ax[5].set_title("2c")

ax[6].imshow(r1.reshape(96,84), cmap="gray")
ax[6].set_title("1c")
print(mu.shape)

# %%
pts = [[3,4],[3,6],[7,6],[6,4]]
cov, (U, VT) = pca(pts)
print(U)

# %% [markdown]
# #### d) Informativeness of each component

# %%
matrix2 = load_images("data/faces/2")
cov, (evec, evals) = dual_pca(matrix2)

mean = matrix2.T.mean(axis=0)
plt.imshow(mean.reshape(96,-1), cmap="gray")

mean_pca = evec.T @ (mean.reshape(-1, 1) - np.mean(matrix2))
print(mean_pca.shape)

# %%
alpha = np.linspace(-10, 10, 10)
k = 3000

for i in range(len(alpha)):
    mean_pca[10] = np.sin(alpha[i]) * k 
    mean_pca[11] = np.cos(alpha[i]) * k
    print(mean_pca[0], mean_pca[1])
    
    avg_face_transformed = evec @ mean_pca + np.mean(mean)
    avg_face_transformed = normalize(avg_face_transformed)
    plt.imshow(avg_face_transformed.reshape((96, 84)), cmap="gray", vmin=0, vmax=255)
    plt.draw()
    plt.pause(1)


# %% [markdown]
# More important eigenvectors have a bigger impact on the image, so changing the weights shows more difference

# %% [markdown]
# ### e) Reconstruction of a foreign image

# %%
elephant = cv2.imread("data/elephant.jpg", 0)
elephant = np.reshape(elephant, (-1, 1))

cov, (evec, evals) = dual_pca(matrix)

projected = evec.T @ (elephant - mu)
reconstructed = evec @ (projected + mu)

fig, ax = plt.subplots(1,3)

ax[0].imshow(elephant.reshape(96,84), cmap="gray")
ax[1].imshow(reconstructed.reshape(96,84), cmap="gray")
ax[2].imshow(img.reshape(96,84), cmap="gray")




