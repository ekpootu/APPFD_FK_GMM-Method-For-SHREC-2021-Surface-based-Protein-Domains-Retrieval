
# This script implements a function to compute FISHER VECTOR for a colection of D-dimensional set of shape descriptors.
'''
This function below is adapted from https://gist.github.com/danoneata/9927923, 
based on the paper: https://hal.inria.fr/file/index/docid/619403/filename/final.r1.pdf

NOTE: Do not forget to L2 and power normalize the Fisher vectors before feeding them into the classifier; 
it has been shown it visibly improves performance, see table 1 in (Sanchez et al., 2013). 
Here is a code example of how the Fisher vectors are prepared for classification by 
normalizing and building the corresponding kernel matrices.
'''

def fisher_vector(xx, gmm):
	"""Computes the Fisher vector on a set of descriptors.
	Parameters
	----------
	xx: array_like, shape (N, D) or (D, )
		The set of descriptors
	gmm: instance of sklearn mixture.GMM object
		Gauassian mixture model of the descriptors.
	Returns
	-------
	fv: array_like, shape (K + 2 * D * K, )
		Fisher vector (derivatives with respect to the mixing weights, means
		and variances) of the given descriptors.
	Reference
	---------
	J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
	Vectors for Image Categorization.  In ICCV, 2011.
	http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
	https://hal.inria.fr/file/index/docid/619403/filename/final.r1.pdf
	"""
	xx = np.atleast_2d(xx)
	N = xx.shape[0]

	# Compute posterior probabilities.
	Q = gmm.predict_proba(xx)  # NxK

	# Compute the sufficient statistics of descriptors.
	Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
	Q_xx = np.dot(Q.T, xx) / N
	Q_xx_2 = np.dot(Q.T, xx ** 2) / N

	# Compute derivatives with respect to mixing weights, means and variances.
	d_pi = Q_sum.squeeze() - gmm.weights_
	#print('\nDerivative w.r.t Weights:\n', np.asarray(d_pi).shape)
	d_mu = Q_xx - Q_sum * gmm.means_
	#print('\nDerivative w.r.t Means:\n', np.asarray(d_mu).shape)
	d_sigma = (
		- Q_xx_2
		- Q_sum * gmm.means_ ** 2
		+ Q_sum * gmm.covariances_
		+ 2 * Q_xx * gmm.means_)
	#print('\nDerivative w.r.t Sigma:\n', np.asarray(d_sigma).shape)

	# Merge derivatives into a vector.
	#	return np.hstack((d_pi, d_mu.flatten()))
	return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))
	
# ------------------------------------------------------------------------------------------------- #
# Source: #https://bitbucket.org/doneata/fv4a/raw/9cd355701c1657eff11a71f8ce4cc42ddd381113/evaluate.py
def compute_L2_normalization(xx):
	""" Computes the L2 norm along the rows, i.e. for each example.

	Input
	-----
	xx: array [N, D]
		Data.

	Output
	------
	Z: array [N]
		Normalization terms.

	"""
	return np.sum(xx ** 2, axis=1)

# Source: #https://bitbucket.org/doneata/fv4a/raw/9cd355701c1657eff11a71f8ce4cc42ddd381113/evaluate.py
def L2_normalize(xx):
	""" L2-normalizes each row of the data xx.

	Input
	-----
	xx: array [N, D]
		Data.

	Output
	------
	yy: array [N, D]
		L2-normlized data.

	"""
	Zx = compute_L2_normalization(xx)
	return xx / np.sqrt(Zx[:, np.newaxis])

# Source: #https://bitbucket.org/doneata/fv4a/raw/9cd355701c1657eff11a71f8ce4cc42ddd381113/evaluate.py
def power_normalize(xx, alpha = 0.5):
	""" Computes a alpha-power normalization for the matrix xx. """
	return np.sign(xx) * np.abs(xx) ** alpha

# ------------------------------------------------------------------------------------------------- #
