# Import needed libraries and packages
import numpy as np
import matplotlib.pyplot as plt

# Import algorithm that computes local keypoints APPFD descriptors as extract
np.set_printoptions(suppress = True)

import ekpoUtilities as ekpo

import open3d
from sklearn import preprocessing
import os
from numpy.core.umath_tests import inner1d

import gc




# -------------------------------------------------------------------------------------- #
'''
THIS FUNCTION IS SUITABLE TO BE USED IN THE IMPLEMENTATION:
KDA (KeypointsDescriptorAggregation), FV(Fisher Vector), GMM(Gaussian Mixture Model), APPFD(Augmented Point Pair Features Descriptor).
'''
#Similar to the SIFT descriptor, for EVERY 'keypoint' or Local Surface Patch - 'LSP', we want to compute and SAVE/OUTPUT a descriptor (i.e. Feature-Vector, FV).
#Code modified on Tuesday, 16th December, 2019 and on Monday, 2nd March, 2021 by Ekpo Otu(eko@aber.ac.uk).
def keypoints_APPFD_6x35bins(pointsCloud, normals, nSamples, r, nBins, voxel_size):
	'''    
	INPUT: 
	i. pointsCloud: N x 3 array, PointsCloud for a single 3D model.

	ii. normals:  N x 3 array of Normal Vectors correspoinding to every points in the pointsCloud (i).

	iii. nSamples - Number 'Random'/'Uniform' points to samples from 3D Triangular Mesh (i.e filename.obj). Default N = 3500

	iv. r (Float: Default = 0.27): Radius param, used by r-nn search to determine the size of Local Surface Patch or Region. 

	v. nBins = 35     #Number of bins for the 1-dimensional histogram of each of the Feature-dimension. Default = 15.

	vi. voxel_size(Float, Default = 0.15):  Parameter to be used by the Voxel Down-Sampling function of Open3D.


	OUTPUT:
	i. lspDescr: An [K x D] array of LSP descriptors for a single 3D object. Where K is the number of keypoints detected (or determined) for the 3D surface, and D is the local APPFD dimension
		which, in this case happens to be 6 x 35 = 210-dimensions.

	Author: Ekpo Otu (eko@aber.ac.uk)
	'''
	# Seed for repeatable results during different runs.
	np.random.seed(231)
	'''
	Perform Voxel-Downsampling on Mesh - Using the Parameter: voxel_size
	'''
	#Step 1: Convert point cloud to .pcd format - Using Open3d
	pcd = open3d.geometry.PointCloud()
	pcd.points = open3d.utility.Vector3dVector(pointsCloud)

	#Step 2: Down-sample the RMS-Scaled Points Cloud data.
	downsampledCloud = ekpo.downsampleCloud_Open3d(pcd, voxel_size)
	print("Downsampled Cloud Size:\t\t", len(downsampledCloud))

	#Step 3: 
	#Find the k-clossest point(s) in the Scaled Cloud to EACH down-sampled Cloud, and REPLACE those in down-sampled cloud, where k = 1.

	# Call function to get the actual Down-sampled cloud that lies originally on the mesh surface. V1 - returns dsPs, dsNs
	actualDScloud, actualDScloudnormals = ekpo.getActual_subCloud_and_normals(downsampledCloud, pointsCloud, normals)
	
	accummulated_keypoints_descriptors = []

	#Loop through each points in Actual Down-sampled pointCloud (actualDScloud). 
	for pnt in range(0, len(actualDScloud)):
		ith_point = actualDScloud[pnt]                 #Interest point - data
		ith_pointNorm = actualDScloudnormals[pnt]      #Interest point - data

		#Local Surface Patch = 'neighbours'
		neighbours, neighbours_normals = ekpo.rnn_normals_skl(pointsCloud, normals, ith_point, r, leafSize = 30)

		#Compute Patch Centre (i.e neighbour-centre)
		patchCentre = np.mean(neighbours, axis = 0)
		#Find location of interest point to patch-centre.
		location = ith_point - patchCentre
		'''
		SD2:
		Compute Surflet-Pair SD for the Local Surface Path (neighbours) - about the ith-Point.
		'''    
		#mini-Step 1: Get corresponding 'normals' to all points in the 'locsal surface patch', and return Surflets.
		#Local Surface Patch Surflets
		localPatch_surflets = ekpo.gsp(neighbours, neighbours_normals)

		#mini-Step 2: Randomly, Select 'all possible combinations of' TWO SURFLETS or 'Surflet-Pairs', i.e (p1, n1) and (p2, n2), from 'surflet' array / list from above.
		localPatch_surflets_pairs = ekpo.gpe(localPatch_surflets, comb = 2)      # comb = possible combinations = 2.

		#mini-Step 3:  compute CANONICAL COORDINATES FROM EACH SURFLETS-PAIR IN STEP 2.
		#From EACH of the Local Patch Surflet-Pairs in (2), compute the constraints in equation (i)

		#All p1
		p1 = localPatch_surflets_pairs[:, 0, 0, :]
		#All p2
		p2 = localPatch_surflets_pairs[:, 1, 0, :]
		#All n1
		n1 = localPatch_surflets_pairs[:, 0, 1, :]
		#All n2
		n2 = localPatch_surflets_pairs[:, 1, 1, :]
		#P2 - P1
		p2_p1 = localPatch_surflets_pairs[:, 1, 0, :] - localPatch_surflets_pairs[:, 0, 0, :]
		#P1 - P2
		p1_p2 = localPatch_surflets_pairs[:, 0, 0, :] - localPatch_surflets_pairs[:, 1, 0, :]
		'''
		COMPUTING FOR LHS constraint
		lhs = abs(n1.dot(p2_p1))
		''' 
		lhs = abs(np.einsum('ij,ij->i', localPatch_surflets_pairs[:, 0, 1, :], (localPatch_surflets_pairs[:, 1, 0, :] - localPatch_surflets_pairs[:, 0, 0, :])))    #Left-Hand-Side
		lhs[np.isnan(lhs)] = 0.
		'''
		COMPUTING FOR RHS constraint
		rhs = abs(n2.dot(p2_p1))
		''' 
		rhs = abs(np.einsum('ij,ij->i', localPatch_surflets_pairs[:, 1, 1, :], (localPatch_surflets_pairs[:, 1, 0, :] - localPatch_surflets_pairs[:, 0, 0, :])))    #Right-Hand-Side
		rhs[np.isnan(rhs)] = 0. 
		# -------------------------------------------------------------------------------------- #
		'''                             LHS COMPUTATIONS                                       '''
		# -------------------------------------------------------------------------------------- #
		vecs1 = p1 - patchCentre
		vecs2 = p1 - location

		#LHS angle
		lhs_angles1 = ekpo.angrw(p1_p2, vecs1)
		lhs_angles2 = ekpo.angrw(p1_p2, vecs2)
		
		crossP1 = np.cross(p2_p1, n1)
		crossP1[np.isnan(crossP1)] = 0.
		
		V1 = ekpo.div0(crossP1, np.sqrt(inner1d(crossP1, crossP1))[:, None])
		
		W1 = np.cross(n1, V1)
		W1[np.isnan(W1)] = 0.
		
		x = np.einsum('ij,ij->i', W1, localPatch_surflets_pairs[:, 1, 1, :])    #x = W1.dot(n2)
		x[np.isnan(x)] = 0.
		y = np.einsum('ij,ij->i', n1, localPatch_surflets_pairs[:, 1, 1, :])    #y = U1.dot(n2)
		y[np.isnan(y)] = 0. 
		'''
		SURFLETS-PAIR 4-D FOR LHS Features
		'''
		alpha1 = np.arctan2(x, y)
		beta1 = np.einsum('ij,ij->i', V1, localPatch_surflets_pairs[:, 1, 1, :]) #V1.dot(n2)
		
		normedP1 = ekpo.div0(p2_p1, np.sqrt(inner1d(p2_p1, p2_p1))[:, None])    
		gamma1 = np.einsum('ij,ij->i', n1, normedP1)
		rheo1 = np.sqrt(inner1d(p2_p1, p2_p1))
		'''
		THE 6-DIMENSIONAL FEATURES-COMBINATION FOR LHS ARE
		'''
		rppf_lhs = np.column_stack((lhs_angles1, lhs_angles2, alpha1, beta1, gamma1, rheo1))
		'''
		IF constraint == TRUE  (Satisfying LSH)
		'''
		#INDEX or INDICES in LHS, where its elements is <= those in RHS
		indx = np.asarray(np.nonzero(lhs <= rhs))
		final_rppf_lhs = np.squeeze(rppf_lhs[indx], axis = 0)
		
		# -------------------------------------------------------------------------------------- #
		'''                             RHS COMPUTATION                                       '''
		# -------------------------------------------------------------------------------------- #
		vecs1x = p2 - patchCentre
		vecs2x = p2 - location
		#vecs3x = p1_p2

		#LHS angle
		rhs_angles1 = ekpo.angrw(p2_p1, vecs1x)
		rhs_angles2 = ekpo.angrw(p2_p1, vecs2x)
		
		crossP2 = np.cross(p1_p2, n2)
		crossP2[np.isnan(crossP2)] = 0.
		
		V2 = ekpo.div0(crossP2, np.sqrt(inner1d(crossP2, crossP2))[:, None])
		
		W2 = np.cross(n2, V2)
		W2[np.isnan(W2)] = 0.
		
		x2 = np.einsum('ij,ij->i', W2, localPatch_surflets_pairs[:, 0, 1, :])    #x2 = W2.dot(n1)
		x2[np.isnan(x2)] = 0.
		y2 = np.einsum('ij,ij->i', n2, localPatch_surflets_pairs[:, 0, 1, :])    #y2 = U2.dot(n1)
		y2[np.isnan(y2)] = 0.
		'''
		SURFLETS-PAIR 4-D FOR RHS Features
		'''
		alpha2 = np.arctan2(x2, y2)
		beta2 = np.einsum('ij,ij->i', V2, localPatch_surflets_pairs[:, 0, 1, :]) #V2.dot(n1)
		
		normedP2 = ekpo.div0(p1_p2, np.sqrt(inner1d(p1_p2, p1_p2))[:, None])
		gamma2 = np.einsum('ij,ij->i', n2, normedP2)
		rheo2 = np.sqrt(inner1d(p1_p2, p1_p2))
		'''
		THE 6-DIMENSIONAL FEATURES-COMBINATION FOR RHS ARE
		'''
		rppf_rhs = np.column_stack((rhs_angles1, rhs_angles2, alpha2, beta2, gamma2, rheo2))
		'''
		IF constraint == FALSE  (Satisfying RSH)
		'''
		#INDEX or INDICES in LHS, where its elements is > those in RHS
		indxx = np.asarray(np.nonzero(lhs > rhs))
		final_rppf_rhs = np.squeeze(rppf_rhs[indxx], axis = 0)

		'''
		OVEARALL, Combine the TWO final features, using Numpy V-Stack()
		'''
		full_final_fppf = np.vstack((final_rppf_lhs, final_rppf_rhs))

		columns_1to5 = preprocessing.minmax_scale(full_final_fppf[:, 0:5])
		column_6 = full_final_fppf[:, 5]

		#Now, use np.column_stack() to form back the 6-features-columns.
		normalizedfeats = np.column_stack((columns_1to5, column_6))
		
		#COMPUTE LSP (keypoint) DESCRIPTOR AT THIS POINT - FOR CURRENT KEYPOINT (lsp).
		# -------------------------------------------------------------------------------------- #
		'''
		Finally, bin EACH feature-dimension of the 'normalizedfeats = []' into a 1-DIMENTIONAL histogram with nBins, and return the CONCATENATED histograms.
		'''
		#BINNING FEATURE-DIM-1 (ANGLE1)
		hist1, _ = np.histogram(normalizedfeats[:, 0], bins = nBins, density = False)
		norm_hist1 = hist1.astype(np.float32) / hist1.sum()
		
		#BINNING FEATURE-DIM-2 (ANGLE2)
		hist2, _ = np.histogram(normalizedfeats[:, 1], bins = nBins, density = False)
		norm_hist2 = hist2.astype(np.float32) / hist2.sum()
		
		#BINNING FEATURE-DIM-3 (ALPHA)
		hist3, _ = np.histogram(normalizedfeats[:, 2], bins = nBins, density = False)
		norm_hist3 = hist3.astype(np.float32) / hist3.sum()
		
		#BINNING FEATURE-DIM-4 (BETA)
		hist4, _ = np.histogram(normalizedfeats[:, 3], bins = nBins, density = False)
		norm_hist4 = hist4.astype(np.float32) / hist4.sum()
	
		#BINNING FEATURE-DIM-5 (GAMMA)
		hist5, _ = np.histogram(normalizedfeats[:, 4], bins = nBins, density = False)
		norm_hist5 = hist5.astype(np.float32) / hist5.sum()
		
		#BINNING FEATURE-DIM-6 (RHO)
		hist6, _ = np.histogram(normalizedfeats[:, 5], bins = nBins, density = False)
		norm_hist6 = hist6.astype(np.float32) / hist6.sum()
		
		#NOW, COMBINE all normalized-1D-Histogram INTO one single Feat-Vec.
		keypoint_APPFD = np.concatenate((norm_hist1, norm_hist2, norm_hist3, norm_hist4, norm_hist5, norm_hist6))
		#Normalize keypoint_APPFD
		keypoint_APPFD = ekpo.normalize1Darrayx(keypoint_APPFD)
		# -------------------------------------------------------------------------------------- #
		accummulated_keypoints_descriptors.append(keypoint_APPFD)

	
	#Vertically STACK all Keypoints Descriptors INTO one LARGE Matrix Array.
	accummulated_keypoints_descriptors = np.asarray(accummulated_keypoints_descriptors)
	gc.collect()

	return accummulated_keypoints_descriptors 
