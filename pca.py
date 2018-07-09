import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def covarience_matrix(X):
    #standardizing data
    X_std = StandardScaler().fit_transform(X)

    #sample means of feature columns' of dataset
    mean_vec = np.mean(X_std, axis=0) 
    #covariance matrix
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    #if right handside is ( Xstd - mean(Xstd) )^T . ( Xstd - mean(Xstd) )
    #simplyfying X^T.X / ( n - 1 )
    cov_mat = np.cov(X_std.T)
    return cov_mat

def max_min_bi_corel(X):
    a = covarience_matrix(X)
    a[a>=1] = 0
    maxcor = np.argwhere(a.max() == a)[0] # reverse 1

    b = covarience_matrix(x)
    mincor = np.argwhere(b.min() == b)[0] # reverse 1

    return maxcor,mincor

def eign_decomposition(cov_mat):
    #eign_decomposition
    eign_vals, eign_vecs = np.linalg.eig(cov_mat)

    print('Eigenvalues \n%s' %eign_vals)
    print('\nEigenvectors \n%s' %eign_vecs)

    
    #for i in range(len(eig_vals)):
    #    (np.abs(eig_vals[i]), eig_vecs[:,i] )
    #make a list of (eigenvalue, eigenvector) tuples
    eign_pairs = [ (np.abs(eign_vals[i]), eign_vecs[:,i] ) for i in range(len(eign_vals)) ]
    return eign_pairs

data = pd.read_csv('iris.csv')
data_features = data.ix[:,:-1]
data_label = data.ix[:,4]

#covariance_matrix
cov_mat = covarience_matrix(data_features)

#eign_decomposition
eig_pairs = eign_decomposition(cov_mat)
#Sorting the (eignvalue, eignvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

#you can select the principle components by eign values, which one can be dropped
#visually confirming about the list is correctly sorted by decreasing eignvaluess
print("Eigenvalues in descending order:")
for i in eig_pairs:
    print(i[0])
    
#projection_matrix of our concatenated top k eigen vectors
matrix_w = np.hstack(( eig_pairs[0][1].reshape(4,1),
                       eig_pairs[1][1].reshape(4,1),
                       eig_pairs[2][1].reshape(4,1) ))

print("Matrix W:\n", matrix_w)
'''
we can choose how many dimensions we want for our subspace by chosing that amount of eignvectors
to construct our dxk dimensional eignvector matrix w. lastly we will use our projection matrix
to transform our samples onto to the subspaces via a simple dot product operation.
'''
