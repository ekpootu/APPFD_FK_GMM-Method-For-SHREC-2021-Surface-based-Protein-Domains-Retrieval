"""
Utility functions needed to compute the APPFD-FK-GMM descriptor for 3D Triangular mesh.

@author: Dr. Ekpo Ekpo Otu (eko@aber.ac.uk)
"""
# ------------------------------------------------------------------------------------------------- #
#Import the needed library(ies).
import numpy as np
import ntpath
import open3d



import glob, re
from scipy.spatial.distance import pdist, squareform 
from scipy.spatial import cKDTree
# ------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------- #
#Ignore wanrnings arising from 'invalid' division and 'division' by 0.
def div0(a, b):
	with np.errstate(divide = 'ignore', invalid = 'ignore'):
		c = np.true_divide(a, b)
		c[ ~ np.isfinite(c)] = 0. 
	return c
	

# Function that computed Alpha and Thetha features for LSP
def angrw(vecs1, vecs2):
	p1 = np.einsum('ij,ij->i', vecs1, vecs2)
	p2 = np.einsum('ij,ij->i', vecs1, vecs1)
	p3 = np.einsum('ij,ij->i', vecs2, vecs2)
	p4 = div0(p1, np.sqrt(p2 * p3))
	return np.arccos(np.clip(p4, -1.0, 1.0))
# ------------------------------------------------------------------------------------------------- #

#Function that TAKES-IN 'Cloud' vs 'Normals' then uses scipy library's cKDTree to find 1-Nearest neighbour to point p of interest
#And returns the nearest POINT and NORMALS
def k_nnVSnormal(points3D, Normals, interestPoint, k):
	'''
	Default value of k = 7.
	from scipy.spatial import cKDTree
	Coded By: Dr. Ekpo Otu (eko@aber.ac.uk)    Tuesday, February 5th, 2019.
	'''
	#Construct a KD-Tree using the scipy library
	tree = cKDTree(points3D, leafsize = 2)
	#Find k nearest indexes in the data. N/B: tree.query returns two values (i.e distance, and index/indices of the k points)
	dist, indx = tree.query(interestPoint, k)
	'''
	dist == Distances between each of the k-Nearest points and the interest-point
	'''
	#Now, retrieve the actual coordinates to those indices
	kClossestCoords = points3D[indx]
	kClossestCoordsNormals = Normals[indx]

	#Return Coordinates of k-Nearest points to interestPoint & Distance
	return kClossestCoords, kClossestCoordsNormals
# ------------------------------------------------------------------------------------------------- #

#Function to Normalize [numpy 1D array].
def normalize1Darrayx(data):
	'''
	NOTE: Adding together the values returned will NOT SUM to 1.
	data - Must be an [N x 1] or [1 x N] array.
	
	Author: Ekpo (eko@aber.ac.uk)  25th February, 2019
	'''
	dataMin = data.min()
	return (data - dataMin) / (data.max() - dataMin) 
	
# ------------------------------------------------------------------------------------------------- #

#Open3d Python Function to Down-sample a Given Point Cloud, using Voxel Grid.
def downsampleCloud_Open3d(pcdcloud, voxel_size = 0.15):
	'''
	INPUTS:
	pcdcloud - Open3D.PointCloud object (pcd), loaded in with Open3d Python.
	voxel_size(Float) - Size of occupied voxel grid. 
	OUTPUTS:
	dspcd_to_numpy - Nx3 array of the down-sampled point cloud. 
	REF:
	Author: Dr. Ekpo Otu, eko@aber.ac.uk   Date: 21th December, 2018
	
	import open3d
	import numpy as np
	'''
	downpcd = pcdcloud.voxel_down_sample(voxel_size)
	dspcd_to_numpy = np.asarray(downpcd.points)
	return dspcd_to_numpy


# ------------------------------------------------------------------------------------------------- #

#Function to find the nearest neighbours within a sphere radius 'r', to the interest point, (using Scikit-Learn) and return those 'neighbours' and 'normals' corresponding to the neighbours.
def rnn_normals_skl(scaledPs, Ns, ip, r, leafSize = 30):
	'''
	INPUT:
	-----
	scaledPs(N x 3 array): Point Cloud data 
	ip(1 x 3 array): A single point from the input 'scaledPs' array.
	r(Float: Default, r=0.17 for Normal Vector Estimation, r=0.35 for Local Surface Patch Selection.): 
		Sphere radius around which nearest points to the 'Interest Points' are obtained. 
	
	OUTPUT:
	-----
	neighbours(N x 3 array): Coordinates of the points neighbourhood within the distance givien by the radius, 'r'.
	dist(1 x N array): Contains the distances to all points which are closer than 'r'. N is the size or number of neighbouring points returned.
	
	Author: Dr. Ekpo Otu(eko@aber.ac.uk) 
	'''	
	neigh = NearestNeighbors(radius = r)
	neigh.fit(scaledPs)
	NearestNeighbors(algorithm = 'auto', leaf_size = leafSize) 
	rng = neigh.radius_neighbors([ip]) 
	dist = np.asarray(rng[0][0]) 
	ind = np.asarray(rng[1][0]) 
	if len(ind) < 5:
		k = 15 
		tree = cKDTree(scaledPs, leafsize = 2) 
		distn, indx = tree.query(ip, k) 
		kcc = scaledPs[indx] 
		nNs = Ns[indx]	 
		return kcc, nNs
	else:
		nn = scaledPs[ind] 
		nNs = Ns[ind] 
		return nn, nNs


# ------------------------------------------------------------------------------------------------- #
# Helper functions
def gsp(Ps, Ns):
	surflets = [(Ps[i], Ns[i]) for i in range(0, len(Ps))]
	return np.asarray(surflets)
	
def gpe(val, comb = 2):
	return np.array(list(combinations(val, comb)))

# ------------------------------------------------------------------------------------------------- #

#Function to find the principal axes of a 3D point cloud.
def computePCA(pointCloud):
	"""
	INPUT: 
		pointCloud -  [N x 3] array of points (Point Cloud) for a single input 3D model.
	OUTPUT:
		eigenvectors - Three floating point values, representing the magnitude of principal directions in each 3-dimensions.
		
	Author: Dr. Ekpo Otu (eko@aber.ac.uk)
	"""
	#Compute the centroid/mean of input point cloud AND Center the point cloud on its centroid/mean
	pointCloud -= np.mean(pointCloud, axis=0)
	
	#Transpose the Centered point cloud, so we now have (3xN) array, instead of (Nx3).
	#pointCloud is transposed due to np.cov/corrcoef syntax
	coords = pointCloud.T
	
	covarianceMatrix = np.cov(coords)
	#covarianceMatrix = coords.T.dot(coords)
	eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)
	
	order = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[order]
	eigenvectors = eigenvectors[:, order]
		
	#Output/Return eigenVectors alone. 
	return eigenvectors

# ------------------------------------------------------------------------------------------------- #

#Function that uses scipy library's 'FASTER' cKDTree to find Nearest neighbour within a radius r, to point p of interest
#With the condition that the minimum number of r-Neighbours to any given 'interestPoint' MUST be greater than 5. Else k-NN,
#where k = 9, is used instead. #def rnn_conditional(points3D, interestPoint, r = 0.17, k = 9):
def rnn_conditional(points3D, interestPoint, r, k):
	'''
	INPUT:
		- points3D: Nx3 Array of pointCloud.
		- interestPoint: 1x3 vector or coordinate, which is EACH point from the 'sub-sampled points' or 'keypoints'.
		- r (Floating Value): Radius, 'r', around which Neighbouring points to 'interestPoint' are to be determined.
				Usually r = 0.02 for pointCloud Normals estimation, or 0.04/0.05 for PFH/Surflet-Pair based SD
		- k (Integer Value): If the 'number of r-neigbours' to 'interestPoint' is < 5, then USE k-NN search on 'points3D', where 'k' = k.
	OUTPUT:
		- rClosestPoints: r-Nearest points to interest point (i), if len(rClosestPoints) >= 5
		- kClosestPoints: k-Nearest points to interest point (i), if len(rClosestPoints) < 5
		
	Author: Dr. Ekpo Otu (eko@aber.ac.uk)
	'''
	#from scipy.spatial import cKDTree
	tree = cKDTree(points3D, leafsize = 2)
	indx = tree.query_ball_point(interestPoint, r)
	rClossestCoords = points3D[indx]
	#print("\nHow Many Neigbhouring Points/Coordinates to interestPoint'{}', When The Value of r = {}:\n".format(interestPoint, r), rClossestCoords.shape[0])
	#print("\nrClossestCoords.shape[0]:\n", rClossestCoords.shape[0])
	
	#CONDITION:
	#Check if the number of neighbours is 5 or more. 5 points should beminimum!
	if rClossestCoords.shape[0] < 5:
		#Then employ k-NN search to enform 5 minimum neighbours. i.e k = 5
		_, indx = tree.query(interestPoint, k)  #Find k nearest indexes in the data.
		#Now, retrieve the actual coordinates to those k-indices
		kClossestCoords = points3D[indx]
		#print("\nHow Many Neigbhouring Points/Coordinates to interestPoint '{}', When The Value of k = {}:\n".format(interestPoint, k), kClossestCoords.shape[0])
		return kClossestCoords
	else:
		return rClossestCoords	
# ------------------------------------------------------------------------------------------------- #

#Function to compute 'Ns' for every p in Ps.
#This function uses scipy library's cKDTree to find k-Nearest neighbours OR points within a radius rr, to point p of interest
def normalsEstimationFromPointCloud_PCA(Ps, rr = 0.17, kk = 9):
	'''
	INPUTS:
		Ps:      N x 3 Array of 'pointCloud', sampled from 3D Triangular Mesh.
		rr (Floating Value): Radius, 'rr', around which Neighbouring points to 'ip' are to be determined.

		kk (Integer Value): The number of neighbours to an interest-point, to use for normal calculation. Default/Minimum = between 7 to 15. Really depends on the number of random points sampled for Ps.	
	OUTPUT:
		[Ps], [Ns]: Each, an N x 3 array of data, representing 'Ps'.
		
	Author: Dr. Ekpo Otu(eko@aber.ac.uk)
	'''
	pts = []
	Ns = []
	for i in range(0, len(Ps)):
		ip = Ps[i]
		pts.append(ip)
		nn = rnn_conditional(Ps, ip, rr, kk)
		evec = computePCA(nn)
		Pn = evec[:, 2]
		Ns.append(Pn)
		
	return np.asarray(pts), np.asarray(Ns)	
# ------------------------------------------------------------------------------------------------- #

#Given a Down-sampled pointsCloud or sub-Cloud, 'Find and Return' the EXACT points from the 'main pointsCloud' that are
#CLOSEST to the sub-Cloud, and the respective/corresponding subCloud Normals.
def getActual_subCloud_and_normals(subCloud, mainCloud, normals):
	'''
	INPUTS:
	subCloud(N x 3 array):  minimal points, Down-sampled from the main pointsCloud. 
	mainCloud(N x 3 array):  main pointsCloud.
	normals(N x 3 array):  Normal Vectors to the mainCloud.
	
	Coded By: Dr. Ekpo Otu (eko@aber.ac.uk)    Tuesday, February 5th, 2019.
	'''
	k = 1
	actual_dsCloud = []  #ds = Down-sampled
	actual_dsCloudNormals = []  #ds = Down-sampled

	for p in subCloud:
		coord, norm = k_nearestNeigbourVSnormal(mainCloud, normals, p, k)
		actual_dsCloud.append(coord)
		actual_dsCloudNormals.append(norm)

	return np.asarray(actual_dsCloud), np.asarray(actual_dsCloudNormals)
# ------------------------------------------------------------------------------------------------- #


