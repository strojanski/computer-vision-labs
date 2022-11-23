import numpy as np
import cv2
from matplotlib import pyplot as plt


def gauss(sigma):
	x = np.arange(np.floor(-3 * sigma), np.ceil(3 * sigma + 1))
	k = np.exp(-(x ** 2) / (2 * sigma ** 2))
	k = k / np.sum(k)
	return np.expand_dims(k, 0)


def gaussdx(sigma):
	x = np.arange(np.floor(-3 * sigma), np.ceil(3 * sigma + 1))
	k = -x * np.exp(-(x ** 2) / (2 * sigma ** 2))
	k /= np.sum(np.abs(k))
	return np.expand_dims(k, 0)


def convolve(I: np.ndarray, *ks):
	"""
	Convolves input image I with all given kernels.

	:param I: Image, should be of type float64 and scaled from 0 to 1.
	:param ks: 2D Kernels
	:return: Image convolved with all kernels.
	"""
	for k in ks:
		k = np.flip(k)  # filter2D performs correlation, so flipping is necessary
		I = cv2.filter2D(I, cv2.CV_64F, k)
	return I


def simple_descriptors(I, Y, X, n_bins = 16, radius = 40, sigma = 2):
	"""
	Computes descriptors for locations given in X and Y.

	I: Image in grayscale.
	Y: list of Y coordinates of locations. (Y: index of row from top to bottom)
	X: list of X coordinates of locations. (X: index of column from left to right)

	Returns: tensor of shape (len(X), n_bins^2), so for each point a feature of length n_bins^2.
	"""

	assert np.max(I) <= 1, "Image needs to be in range [0, 1]"
	assert I.dtype == np.float64, "Image needs to be in np.float64"

	g = gauss(sigma)
	d = gaussdx(sigma)

	Ix = convolve(I, g.T, d)
	Iy = convolve(I, g, d.T)
	Ixx = convolve(Ix, g.T, d)
	Iyy = convolve(Iy, g, d.T)

	mag = np.sqrt(Ix ** 2 + Iy ** 2)
	mag = np.floor(mag * ((n_bins - 1) / np.max(mag)))

	feat = Ixx + Iyy
	feat += abs(np.min(feat))
	feat = np.floor(feat * ((n_bins - 1) / np.max(feat)))

	desc = []

	for y, x in zip(Y, X):
		miny = max(y - radius, 0)
		maxy = min(y + radius, I.shape[0])
		minx = max(x - radius, 0)
		maxx = min(x + radius, I.shape[1])
		r1 = mag[miny:maxy, minx:maxx].reshape(-1)
		r2 = feat[miny:maxy, minx:maxx].reshape(-1)

		a = np.zeros((n_bins, n_bins))
		for m, l in zip(r1, r2):
			a[int(m), int(l)] += 1

		a = a.reshape(-1)
		a /= np.sum(a)

		desc.append(a)

	return np.array(desc)


def display_matches(I1, pts1, I2, pts2):
	"""
	Displays matches between images.

	I1, I2: Image in grayscale.
	pts1, pts2: Nx2 arrays of coordinates of feature points for each image (first columnt is x, second is y coordinates)
	"""

	assert I1.shape[0] == I2.shape[0] and I1.shape[1] == I2.shape[1], "Images need to be of the same size."

	I = np.hstack((I1, I2))
	w = I1.shape[1]
	plt.imshow(I, cmap='gray')

	for p1, p2 in zip(pts1, pts2):
		x1 = p1[0]
		y1 = p1[1]
		x2 = p2[0]
		y2 = p2[1]
		plt.plot(x1, y1, 'bo', markersize=3)
		plt.plot(x2 + w, y2, 'bo', markersize=3)
		plt.plot([x1, x2 + w], [y1, y2], 'r', linewidth=.8)

	plt.show()