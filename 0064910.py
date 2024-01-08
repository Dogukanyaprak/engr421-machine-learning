#!/usr/bin/env python
# coding: utf-8

# In[35]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats


# In[36]:


group_means = np.array([[-6.0, -1.0],
                        [-3.0, +2.0],
                        [+3.0, +2.0],
                        [+6.0, -1.0]])


# In[37]:


group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +4.0]],
                              [[+2.4, -2.0],
                               [-2.0, +2.4]],
                              [[+2.4, +2.0],
                               [+2.0, +2.4]],
                              [[+0.4, +0.0],
                               [+0.0, +4.0]]])


# read data into memory

# In[38]:


data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")


# get X values

# In[39]:


X = data_set[:, [0, 1]]


# set number of clusters

# In[40]:


K = 4


# STEP 2<br>
# should return initial parameter estimates<br>
# as described in the homework description

# In[41]:


def initialize_parameters(X, K):
    # your implementation starts below
    
    centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter = ",")
    
    distance = dt.cdist(X, centroids)
    
    membership = np.argmin(distance, axis=1)

    means = np.array([np.mean(X[membership == i], axis=0) for i in range(K)])

    covariances = np.array([np.cov(X[membership == m].T) for m in range(K)])

    priors = np.array([np.mean(membership == k) for k in range(K)])

    # your implementation ends above
    return(means, covariances, priors)


# In[42]:


means, covariances, priors = initialize_parameters(X, K)


# STEP 3<br>
# should return final parameter estimates of<br>
# EM clustering algorithm

# In[43]:


def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    
    n_samples, n_features = X.shape
    
    num_iteration = 100
    curr_iteration = 1
    
    while(curr_iteration<=num_iteration):
        
        resp = np.zeros((n_samples, K))
        for k in range(K):
            resp[:, k] = priors[k] * stats.multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
        resp /= np.sum(resp, axis=1, keepdims=True)

        for k in range(K):
            sum_k = np.sum(resp[:, k])
            means[k] = np.sum(resp[:, k][:, np.newaxis] * X, axis=0) / sum_k
            covariances[k] = np.dot((resp[:, k][:, np.newaxis] * (X - means[k])).T, (X - means[k])) / sum_k
            priors[k] = sum_k / n_samples
            
        curr_iteration = curr_iteration + 1
        
    assignments = np.argmax(resp, axis=1)
    
    # your implementation ends above
    return(means, covariances, priors, assignments)


# In[44]:


means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)


# STEP 4<br>
# should draw EM clustering results as described<br>
# in the homework description

# In[51]:


def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    plt.figure(figsize=(8, 6))

    
    colors = ['red', 'blue', 'green','purple']
    
    for i in range(K) :
        c_points = X[assignments == i]
        plt.scatter(c_points[:, 0], c_points[:, 1], color=colors[i], label=f'Cluster {i + 1}')

    
    x, y = np.meshgrid(np.linspace(-8, 8, 200), np.linspace(-8, 8, 200))
    
    for m in range(K) :
        org_density = stats.multivariate_normal.pdf(np.dstack((x, y)), mean=group_means[m], cov=group_covariances[m])
        plt.contour(x, y, org_density, levels=[0.01], colors='black', linestyles='dashed', linewidths=2)

    
    for k in range(K) :
        est_density = stats.multivariate_normal.pdf(np.dstack((x, y)), mean=means[k], cov=covariances[k])
        plt.contour(x, y, est_density, levels=[0.01], colors=colors[k], linestyles='solid', linewidths=2)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)


# In[ ]:




