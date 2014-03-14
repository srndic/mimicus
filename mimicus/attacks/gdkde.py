'''
Copyright 2013, 2014 Pavel Laskov, Nedim Srndic, University of Tuebingen

This file is part of Mimicus.

Mimicus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Mimicus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Mimicus.  If not, see <http://www.gnu.org/licenses/>.
##############################################################################
gdkde.py

Implementation of the Gradient Descent and Kernel Density Estimation 
(GD-KDE) attack.

Created on July 29, 2013.
'''

from math import exp
import sys

import numpy
from numpy import absolute, empty, dot, reshape, where, inner
from numpy.linalg import norm
from numpy.matlib import repmat

from mimicus.tools import featureedit

def gradient_kde(xc, x, y, h):
    '''
    Returns the gradient of the kernel density estimation.
    '''
    neg_idx = where(y == 0.0)[0]
    x_neg = x[neg_idx]
    n = x_neg.shape[0]
    
    kval = empty((n, 1))
    gval = empty(x_neg.shape)
    dist = repmat(xc, n, 1) - x_neg
    for i in xrange(n):
        kval[i] = exp(-sum(absolute(dist[i, :])) / h)
        gval[i,:] = -kval[i] * dist[i, :] / h 
    
    g = -sum(gval) / n
    
    return reshape(g, (1, len(g)))

def gradient_svm(xc, svm):
    '''
    Returns the gradient of the svm.
    '''
    # Computes the value and the gradients (w.r.t. x) of the linear kernel
    def dk_linear(x, y):
        k = inner(x, y)
        dk = y
        return (k, dk)

    # Computes the value and the gradient (w.r.t. x) of the RBF kernel
    def dk_rbf(x, y, gamma):
        k = exp(-gamma / 2 * inner((x - y), (x - y)))
        dk = gamma * k * (y - x)
        return (k, dk)
    
    # Retrieve the internal parameters from the SVM 
    alpha = svm.dual_coef_
    svs = svm.support_vectors_
    nsv = svs.shape[0]
    b = svm.intercept_
    if svm.classes_[0] == 1:
        sign = 1
    else:
        sign = -1
    
    # Compute the kernel row matrix and kernel gradients for xc 
    kval = empty(alpha.shape).T
    kgrad = empty(svs.shape)
    for i in xrange(nsv):
        if svm.kernel == 'linear':
            (k, dk) = dk_linear(xc, svs[i, :])
        elif svm.kernel == 'rbf':
            (k, dk) = dk_rbf(xc, svs[i, :], svm.gamma)
        else:
            raise StandardError("unsupported kernel'" + svm.kernel + "'")
        kval[i] = k
        kgrad[i, :] = dk
    
    retval = sign * dot(alpha, kgrad)
    return retval[0]

def gdkde(svm, fname, scaler, x, y, kde_reg, kde_width, step, max_iter,
          stop_at_boundary=True, verbose=False):
    fseq = numpy.ndarray(shape=(0, 1))
    xseq = numpy.ndarray(shape=(0, x.shape[1]))
    gseq = numpy.ndarray(shape=(0, x.shape[1]))
    mseq = numpy.ndarray(shape=(0, x.shape[1]))
    
    fedit = featureedit.FeatureEdit(fname)
    xc = fedit.retrieve_feature_vector_numpy()
    xc_scaled = scaler.transform(xc, copy=True)
    fseq = numpy.append(fseq, svm.decision_function(xc_scaled))
    idx_null = where(featureedit.FeatureDescriptor.get_feature_mask_numpy() 
                     == 0)[0]
    
    for iteration in range(max_iter):
        # Compute the new feature vector using the SVM and the KDS gradients
        grad_s = gradient_svm(xc_scaled, svm)
        grad_d = gradient_kde(xc_scaled, x, y, kde_width)
        grad = grad_s + kde_reg * grad_d
        full_grad = grad.copy()
        
        grad = grad / norm(grad)
        grad[:, idx_null] = 0
        xc_scaled = xc_scaled - step * grad
        
        fseq = numpy.append(fseq, svm.decision_function(xc_scaled))
        xseq = numpy.append(xseq, xc_scaled, axis=0)
        gseq = numpy.append(gseq, full_grad, axis=0)
        
        if verbose: sys.stdout.write("score[{0}]: {1}\n"
                                        .format(iteration, fseq[iteration]))
        
        if (stop_at_boundary and fseq[len(fseq) - 1] < 0):
            break
    
    xc = featureedit.features_inverse_standardize(list(xc_scaled[0]), scaler)
    reply = fedit.modify_file(xc, verbose=verbose)
    final_xc_scaled = scaler.transform(reply['feats'], copy=True)
    fseq = numpy.append(fseq, svm.decision_function(final_xc_scaled))
    return (xseq, fseq, gseq, mseq, reply['path'])
