"""
Utility functions needed to compute the APPFD-FK-GMM descriptor for 3D Triangular mesh.

@author: Ekpo Ekpo Otu (eko@aber.ac.uk)
"""
# ------------------------------------------------------------------------------------------------- #
#Import the needed library(ies).
import numpy as np
import ntpath
import open3d



import glob, re
from scipy.spatial.distance import pdist, squareform 

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

def getBasename(filename_and_path):
	return ntpath.basename(filename_and_path)
	
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
	Author: Ekpo, eko@aber.ac.uk   Date: 21th December, 2018
	
	import open3d
	import numpy as np
	'''
	downpcd = open3d.voxel_down_sample(pcdcloud, voxel_size)
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
	
	Author: Ekpo Otu(eko@aber.ac.uk) 
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

#Function to compute 'Ns' for every p in Ps.
#This function uses scipy library's cKDTree to find k-Nearest neighbours OR points within a radius rr, to point p of interest
def normalsEstimationFromPointCloud_PCA(Ps, rr = 0.17, kr = 9):
	'''
	INPUTS:
		Ps:      N x 3 Array of 'pointCloud', sampled from 3D Triangular Mesh.
		rr (Floating Value): Radius, 'rr', around which Neighbouring points to 'ip' are to be determined.

		kk (Integer Value): The number of neighbours to an interest-point, to use for normal calculation. Default/Minimum = between 7 to 15. Really depends on the number of random points sampled for Ps.	
	OUTPUT:
		[Ps], [Ns]: Each, an N x 3 array of data, representing 'Ps'.
		
	Author: Ekpo Otu(eko@aber.ac.uk)
	'''
	pts = []
	Ns = []
	for i in range(0, len(Ps)):
		ip = Ps[i]
		pts.append(ip)
		nn = rnn_conditional(Ps, ip, rr, kk)
		_, evec = computePCA(nn)
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
	
	Coded By: Ekpo (eko@aber.ac.uk)    Tuesday, February 5th, 2019.
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


