import numpy as np
 
def PCA(X_scaled , num_components):
    
    #Step-1
    cov_mat = np.cov(X_scaled , rowvar = False)
     
    #Step-2
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-3
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-3
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-4
    X_reduced = np.dot(eigenvector_subset.transpose(), X_scaled.transpose()).transpose()
     
    return X_reduced
