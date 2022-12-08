import numpy as np
import cv2
from matplotlib import pyplot as plt

def normalize_points(P):
	# P must be a Nx2 vector of points
	# first coordinate is x, second is y

	# returns: normalized points in homogeneous coordinates and 3x3 transformation matrix

	mu = np.mean(P, axis=0) # mean
	scale = np.sqrt(2) / np.mean(np.sqrt(np.sum((P-mu)**2,axis=1))) # scale
	T = np.array([[scale, 0, -mu[0]*scale],[0, scale, -mu[1]*scale],[0,0,1]]) # transformation matrix
	P = np.hstack((P,np.ones((P.shape[0],1)))) # homogeneous coordinates
	res = np.dot(T,P.T).T
	return res, T

def draw_epiline(l,h,w):
	# l: line equation (vector of size 3)
	# h: image height
	# w: image width

	x0, y0 = map(int, [0, -l[2]/l[1]])
	x1, y1 = map(int, [w-1, -(l[2]+l[0]*w)/l[1]])

	plt.plot([x0,x1],[y0,y1],'r')

	plt.ylim([0,h])
	plt.gca().invert_yaxis()