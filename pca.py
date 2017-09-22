#get package
import numpy as np
from numpy import linalg as LA
from scipy import stats
import matplotlib.pyplot as plt
import random
import pandas as pd
from pandas import DataFrame, read_csv
from sklearn.decomposition import PCA
import csv
import sys

#soln1: use pandas package
#import data
df_1 = pd.read_csv('dataset_1.csv', sep=',', header=0)
#print(df_1)
#mean-center data
df_2 = df_1.values
row_mean = np.mean(df_2, axis=0)#column means
#print(row_mean)
df_2_c = df_2 - row_mean

#use PCA package
pca_1 = PCA(n_components=2)
pca_1.fit(df_2_c)
eigenvector_1 = pca_1.components_
eigenvalue_1 = pca_1.explained_variance_
#print(eigenvalue_1)
#print(eigenvector_1)
#projection matrix
proj  = pca_1.fit_transform(df_2_c)
#plot the first two pc
plt.scatter(proj[:,0], proj[:,1])
plt.show()
'''

#soln2: use numpy
#1: read data into matrix, ignoring the header
df_2 = np.genfromtxt('dataset_1.csv', delimiter=',', dtype=float)[1:]
#print(df_2)

'''
#2:center the data
x = df_2[:,0]
y = df_2[:,1]
z = df_2[:,2]
#transpose the matrix to find covariance matrix
df_3 = df_2.transpose()
print(df_3)
cov_df_3 = np.cov(df_3)
print(cov_df_3)
'''

#center the data
row_mean = np.mean(df_2, axis=0)#column means
#print(row_mean)
df_2_c = df_2 - row_mean
#print(df_2_c)

#transpose the matrix to get cov matrix
df_3_c = df_2_c.transpose()
cov_df_3_c = np.cov(df_3_c)
print(cov_df_3_c)

#cal eigenvalue and eigenvector
eiva,eive = LA.eig(cov_df_3_c)
print(eiva)
print(eive)#column vectors are eigenvectors

#sort eigenpairs in descending order
#1:make a list of (eigenvalue,eigenvector) tuples
eig_pairs = [(np.abs(eiva[i]), eive[:,i])
             for i in range(len(eiva))]
#2:sort the tupels from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
#3:print the sorted dicreasing eigenvalues to check
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

#projection matrix
list_a = []
for i in range(len(eig_pairs)):
    list_a.append(eig_pairs[i][1])
pro_matrix = np.array(list_a).T
print(pro_matrix)

#calculate pc
pc = np.matmul(df_2_c, pro_matrix)
pc_12 = pc[:,0:2]#get the first two projection clm vectors
print(pc_12)
#print(pca)
#print(df_2_c.shape)#check shape
plt.scatter(pc_12[:,0], pc_12[:,1])
plt.show()
'''
