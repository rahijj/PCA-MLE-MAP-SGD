# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import norm
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt 
import matplotlib.tri as tri
from matplotlib.patches import Ellipse 
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.animation as animation 
from matplotlib.colors import ListedColormap
from numpy import cov
import tensorflow as tf
from statistics import stdev 
import csv

"""Q1) Part A & B."""

def ellipse_1SD(data1, data2, cov1, cov2, u):
  
  plt.figure(0)
  ax = plt.gca()
  plt.scatter(data1[0], data1[1])
  ax.add_patch(Ellipse([0, 0], 2, 2, angle=0, fc = 'None', edgecolor = 'red'))
  plt.show()

  plt.figure(1)
  ax = plt.gca()
  plt.scatter(data2[0], data2[1])
  np.linalg.eig(cov2)

  print ("Eigenvector/Eigenvalues:\n ", np.linalg.eig(cov2))
  eign = np.linalg.eig(cov2)

  # Calculated Angle of rotation for ellipse
  
  angle = (np.arctan2(eign[1][1][1], eign[1][0][1])* 180 / np.pi)
  print("Angle = ",angle)

  ax.add_patch(Ellipse([0, 0], 2*(math.sqrt(eign[0][1])), 2*(math.sqrt(eign[0][0])), angle = angle, fc = 'None', edgecolor = 'red'))
  plt.show()

# Given Covariances

cov1 = np.array([[1, 0], [0, 1]])
cov2 = np.array([[2, -2], [-2, 3]])

# Mean taken as Zero

u = [0, 0]

# Generated Multivariate normal data

data1 = np.random.multivariate_normal(u, cov1, 1000).T
data2 = np.random.multivariate_normal(u, cov2, 1000).T

ellipse_1SD(data1, data2, cov1, cov2, u)

"""Q1) Part C"""

# This function calculates the covariance 

def covar(a, b):
  a_m = a.mean()
  b_m = b.mean()  
  # As we calculate sample covariance, so we divide by (len(a)-1) instead of len(a). 
  res = np.dot((a - a_m), (b - b_m))/(len(a) - 1)
  return res

data1_cov = np.array([[covar(data1[0], data1[0]), covar(data1[0], data1[1])],[covar(data1[1], data1[0]), covar(data1[1], data1[1])]])
data2_cov = np.array([[covar(data2[0], data2[0]), covar(data2[0], data2[1])],[covar(data2[1], data2[0]), covar(data2[1], data2[1])]])

print ("data1 Covariance: ")
print (data1_cov, '\n')

print ("data2 Covariance: ")
print (data2_cov)

"""Q1) Part D

The difference in the Covariances is as a result of the difference in the Global and Sample Dataset. The covariance will be the same when the sample data is equal to the population data.

Q2) Part A
"""

# This function plots the ellipse and mean

def mean_cont(cov, mean, ax):

  plt.plot(mean[0], mean[1], "+r")
  eign = np.linalg.eig(cov)
  angle = (np.arctan2(eign[1][1][1], eign[1][0][1])* 180 / np.pi)
  ax.add_patch(Ellipse([mean[0], mean[1]], 2*(math.sqrt(eign[0][1])), 2*(math.sqrt(eign[0][0])), angle = angle, fc = 'None', edgecolor = 'red'))
  ax.autoscale_view()

def MAP(grid):

  res = []
  
  # Priori Probabilities of the three classes
  
  P_C1 = 0.2
  P_C2 = 0.7
  P_C3 = 0.1

  # Mean of the three classes
  
  u2_1 = [3, 2]
  u2_2 = [5, 4]
  u2_3 = [2, 5]

  # Covariance of the three classes
  
  cov2_1 = np.array([[1, -1], [-1, 2]])
  cov2_2 = np.array([[1, -1], [-1, 2]])
  cov2_3 = np.array([[0.5, 0.5], [0.5, 3]])

  # Discriminant function for the MAP classifier
  
  for i in grid:
    g_x1 = -0.5 * np.dot(np.dot((i-u2_1).T, inv(cov2_1)), (i-u2_1)) - 0.5 * np.log(norm(cov2_1)) + np.log(P_C1)
    g_x2 = -0.5 * np.dot(np.dot((i-u2_2).T, inv(cov2_2)), (i-u2_2)) - 0.5 * np.log(norm(cov2_2)) + np.log(P_C2)
    g_x3 = -0.5 * np.dot(np.dot((i-u2_3).T, inv(cov2_3)), (i-u2_3)) - 0.5 * np.log(norm(cov2_3)) + np.log(P_C3)

    # Choose the class with the highest value.
    
    res.append(np.argmax([g_x1, g_x2, g_x3]))

  return np.array(res)

def ML(grid):

  res = []
  
  # Priori Probabilities of the three classes
  
  P_C1 = 0.2
  P_C2 = 0.7
  P_C3 = 0.1

  # Mean of the three classes
  
  u2_1 = [3, 2]
  u2_2 = [5, 4]
  u2_3 = [2, 5]

  # Covariance of the three classes
  
  cov2_1 = np.array([[1, -1], [-1, 2]])
  cov2_2 = np.array([[1, -1], [-1, 2]])
  cov2_3 = np.array([[0.5, 0.5], [0.5, 3]])

  # Discriminant function for the MAP classifier
  
  for i in grid:
    g_x1 = -0.5 * np.dot(np.dot((i-u2_1).T, inv(cov2_1)), (i-u2_1)) - 0.5 * np.log(norm(cov2_1))
    g_x2 = -0.5 * np.dot(np.dot((i-u2_2).T, inv(cov2_2)), (i-u2_2)) - 0.5 * np.log(norm(cov2_2))
    g_x3 = -0.5 * np.dot(np.dot((i-u2_3).T, inv(cov2_3)), (i-u2_3)) - 0.5 * np.log(norm(cov2_3)) 

    # Choose the class with the highest value.
    
    res.append(np.argmax([g_x1, g_x2, g_x3]))
    
  return np.array(res)

def Ml_Map_Bound():

  cMap = ListedColormap(['green', 'blue','yellow'])
  lab = np.array(["Class 1", "Class 2", "Class 3"])
  
  # Covariance of the three classes
  
  cov2_1 = np.array([[1, -1], [-1, 2]])
  cov2_2 = np.array([[1, -1], [-1, 2]])
  cov2_3 = np.array([[0.5, 0.5], [0.5, 3]])

  # Mean of the three classes
  
  u2_1 = [3, 2]
  u2_2 = [5, 4]
  u2_3 = [2, 5]

  # Meshgrid Dimensions
  
  x_min = 0
  x_max = 7
  y_min = 0
  y_max = 8

  # Meshgrid stepsize
  
  s = 0.05

  xx, yy = np.meshgrid(np.arange(x_min, x_max, s),np.arange(y_min, y_max, s))
  grid = np.c_[xx.ravel(), yy.ravel()]

  Map_r = MAP(grid).reshape(xx.shape)
  Ml_r = ML(grid).reshape(xx.shape)

  fig, ax = plt.subplots(1)
  
  # Creating Proxy Artist for legend
  
  g = mpatches.Patch(color='green', label= lab[0])
  b = mpatches.Patch(color='blue', label= lab[1])
  y = mpatches.Patch(color='yellow', label= lab[2])
  ax.legend(handles=[g,b,y])

  ax.pcolormesh(xx, yy, Map_r, cmap = cMap)
  ax.contour(xx, yy, Map_r)
  mean_cont(cov2_1,u2_1, ax)
  mean_cont(cov2_2,u2_2, ax)
  mean_cont(cov2_3,u2_3, ax)
  plt.title("MAP Classifier")
  plt.show()
 
  fig, ax = plt.subplots(1)
  plt.pcolormesh(xx, yy, Ml_r, cmap = cMap, label = lab)
  plt.contour(xx, yy, Ml_r)
  ax.legend(handles=[g,b,y])

  mean_cont(cov2_1,u2_1, ax)
  mean_cont(cov2_2,u2_2, ax)
  mean_cont(cov2_3,u2_3, ax)
  plt.title("ML Classifier")
  plt.show()

Ml_Map_Bound()

"""Q2) Part B"""

def Conf_Matrix():
  
  cMap = ListedColormap(['green', 'blue','yellow'])
  cMap1 = ListedColormap(['green', 'blue','yellow', 'red'])
  cMap2 = ListedColormap(['red'])


  lab = np.array(["Class 1", "Class 2", "Class 3"])
  lab1 = np.array(["Class 1", "Class 2", "Class 3", "Miss-Classified"])

  
  # Covariance of the three classes
  
  cov2_1 = np.array([[1, -1], [-1, 2]])
  cov2_2 = np.array([[1, -1], [-1, 2]])
  cov2_3 = np.array([[0.5, 0.5], [0.5, 3]])

  # Mean of the three classes
  
  u2_1 = [3, 2]
  u2_2 = [5, 4]
  u2_3 = [2, 5]

  # Priori Probabilities of the three classes
  
  P_C1 = 0.2
  P_C2 = 0.7
  P_C3 = 0.1
  
  data_p1 = 3000*P_C1
  data_p2 = 3000*P_C2
  data_p3 = 3000*P_C3

  # Generating data points.
  
  data2_1 = np.random.multivariate_normal(u2_1, cov2_1, int(data_p1))
  data2_2 = np.random.multivariate_normal(u2_2, cov2_2, int(data_p2))
  data2_3 = np.random.multivariate_normal(u2_3, cov2_3, int(data_p3))

  # Generating actual labels for data.
  
  actual = np.array([np.concatenate([[0] * int(data_p1), [1] * int(data_p2), [2] * int(data_p3)])]).T 
  grid  = (np.concatenate((data2_1, data2_2, data2_3), axis = 0))

  # Passing points to the classifiers that then returns their assigned labels.
  
  Map_r = MAP(grid)
  Ml_r = ML(grid)

  # Generating confusin Matrix
  
  conf_MAP = confusion_matrix(actual, Map_r)
  conf_Ml = confusion_matrix(actual, Ml_r)

  print("\nMAP Confusion Matrix")
  print(100 * (1 - (np.trace(conf_MAP)/np.sum(conf_MAP))), "%")
  print(conf_MAP, "\n")
  print("Ml Confusion Matrix")
  print(100 * (1 - (np.trace(conf_Ml)/np.sum(conf_Ml))), "%")

  print(conf_Ml)

  # Creating Proxy Artist for legend
  
  g = mpatches.Patch(color='green', label= lab[0])
  b = mpatches.Patch(color='blue', label= lab[1])
  y = mpatches.Patch(color='yellow', label= lab[2])
  r = mpatches.Patch(color='red', label= lab1[3])


  test = actual[:,0] == Map_r
  test1 = actual[:,0] == Ml_r
  miss_class = np.where(test == False)
  miss_class1 = np.where(test1 == False)
  Map_r1 = Map_r
  Map_r1[miss_class] = 3

  Ml_r1 = Ml_r
  Ml_r1[miss_class1] = 3


  plt.figure(1, figsize=(20, 12))
  ax = plt.subplot(2,3,1 )
  ax.legend(handles=[g,b,y])
  a = plt.scatter((grid.T)[0], (grid.T)[1], c=actual[:,0], cmap = cMap)
  plt.title("Actual")

  ax = plt.subplot(2,3,2)
  ax.legend(handles=[g,b,y])
  plt.scatter((grid.T)[0], (grid.T)[1], c=Map_r, cmap = cMap)
  mean_cont(cov2_1,u2_1, ax)
  mean_cont(cov2_2,u2_2, ax)
  mean_cont(cov2_3,u2_3, ax)
  plt.title("MAP Classifier")

  ax = plt.subplot(2,3,3)
  ax.legend(handles=[g,b,y])
  plt.scatter((grid.T)[0], (grid.T)[1], c=Ml_r, cmap = cMap)
  mean_cont(cov2_1,u2_1, ax)
  mean_cont(cov2_2,u2_2, ax)
  mean_cont(cov2_3,u2_3, ax)
  plt.title("ML Classifier")

  ax = plt.subplot(2,3,4)
  ax.legend(handles=[r])
  plt.scatter((grid.T)[0][miss_class], (grid.T)[1][miss_class], c=Map_r1[miss_class], cmap = cMap2)
  mean_cont(cov2_1,u2_1, ax)
  mean_cont(cov2_2,u2_2, ax)
  mean_cont(cov2_3,u2_3, ax)
  plt.title("MAP Miss-Classified " + str(len(miss_class[0])) + " Points" )

  ax = plt.subplot(2,3,5)
  ax.legend(handles=[r])
  plt.scatter((grid.T)[0][miss_class1], (grid.T)[1][miss_class1], c=Ml_r1[miss_class1], cmap = cMap2)
  mean_cont(cov2_1,u2_1, ax)
  mean_cont(cov2_1,u2_1, ax)
  mean_cont(cov2_2,u2_2, ax)
  mean_cont(cov2_3,u2_3, ax)
  plt.title("ML Miss-Classified " + str(len(miss_class1[0])) + " Points")

  plt.show()

Conf_Matrix()

"""Q3) Part A"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_flat = []

# Flattening Imgs array so PCA can run on it.

for i in x_train:
  x_train_flat.append(i.flatten())
x_train_flat = np.array(x_train_flat)

print("Original Shape ", x_train.shape)
print("Flattened Shape ", x_train_flat.shape)

def Pca(imgs, d):
  
  print("PCA(" + str(d) + ") Running...")
  
  # Taking transpose to convert to form (Number_of_Samples x Number_of_dimensions)
  
  N_D = imgs.T
  u = np.mean(N_D, axis = 0)
  
  # Minus mean to center data
  
  cent_data = N_D - u
  
  # Generate Covariance matrix
  
  covmat = np.cov(cent_data.T)
  
  # Generate Eigenvalue & Eigenvectors
  
  eig_vals, eig_vecs = np.linalg.eig(covmat)
  
  # Sort the Eigen Values in descending order of Indexes
  
  sorted_eigval = np.argsort(eig_vals)[::-1]
  
  # Select First d Principal Components
  
  pc = eig_vecs.T[sorted_eigval[0:d]]
  
  # Transform Data to Lower Dimension
  
  trans_data = np.dot(cent_data, pc.T).T
  
  return (trans_data, pc, u)

"""Q3) Part B"""

# This function calculates the proportion of variance and 
# ouputs the number of dimensions needed for POV >= 0.95 
def POV(imgs):

  covmat = np.cov(imgs)
  eig_vals, eig_vecs = np.linalg.eig(covmat)
  sorted_eigval = np.argsort(eig_vals)[::-1]
  T = np.sum(eig_vals)
  t = 0
  for d, i in enumerate(sorted_eigval):
    t = t + eig_vals[sorted_eigval[i]]
    check = t/T
    if check >= 0.95:
      print ("Suitable d =",d+1)
      return (d+1)

POV(x_train_flat.T)

"""Q3) Part C"""

# Function Reconstructs using the Pricipal components and mean

def reconstruct(imgs, pc, u):

  print("Reconstructing....")
  rec = np.dot(pc.T, imgs).T + u
  return rec.T

# Function plots MSE against Num of d dimensions chosen

def pca_plot(x_train_flat):

  d = np.arange(1,785,50)
  print ("d =", d)
  err = []
  err2 = []
  for i in d:
    comp_img, pc, u = Pca(x_train_flat.T, i)
    rec = reconstruct(comp_img, pc, u)
    mse = mean_squared_error(x_train_flat.T, rec)
    err.append(mse)
  plt.plot(d,err)
  plt.ylabel("MSE")
  plt.xlabel("Principal Components")
  plt.show()

pca_plot(x_train_flat)

"""Q3) Part D"""

# Function reconstructs the number 8 using differnet number of principal components

def reconstruct_8(x_train_flat):

  d = [1, 10, 50, 250, 784]
  plt.figure(figsize=(10, 10))

  for i, j in enumerate(d):
    rec_shaped = []
    comp_img, pc, u = Pca(x_train_flat.T, j)
    rec = reconstruct(comp_img, pc, u)
    rec_T = rec.T
    for k in rec_T:
      temp = k.reshape(-1,28)
      rec_shaped.append(temp)
    rec_shaped = np.array(rec_shaped)
    image_index = 7777
    plt.subplot(2, 3, i+1)
    plt.imshow(rec_shaped[image_index], cmap='Greys')
    plt.title("d = "+ str(j))
  plt.show()
reconstruct_8(x_train_flat)

"""Q3) Part E"""

# Function that plots eigenvalues against d dimensions

def eigen_plot(imgs):

  d = np.arange(1,785,1)
  N_D = imgs.T
  u = np.mean(N_D, axis = 0)
  cent_data = N_D - u
  covmat = np.cov(cent_data.T)
  eig_vals, eig_vecs = np.linalg.eig(covmat)
  sorted_eigval = np.sort(eig_vals)[::-1]
  plt.plot(d,sorted_eigval)
  plt.ylabel("Eigen Values")
  plt.xlabel("d value")
  plt.show()

eigen_plot(x_train_flat.T)

f = open("dataset3.txt")
dataset3 = []
temp = []

for row in csv.reader(f):
  temp.append(float(row[0]))
  temp.append(float(row[1]))
  temp.append(int(row[2]))
  dataset3.append(temp)
  temp = []

dataset3 = np.array(dataset3)

"""Q4) Part A"""

# Cost Function for Logistic Regression

def cost_func(h, x, y):

  m = x.shape[0]
  c = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
  return c

"""Q4) Part B, C, D, E, F"""

from copy import deepcopy
def sigmoid(par):

  s = 1/(1 + np.exp(-par))
  return s 

def gradient(h, x, y):

  m = x.shape[0]
  g = (1/m) * np.dot((h - y), x)
  return g 

def accuracy(s_list, y):

  s_list = np.array(s_list).flatten()
  y = y.astype(int)
  pred = (s_list >= 0.5).astype(int)
  return ((np.sum(pred == y)/len(pred))*100)

def SGD(dataset3): 

  no_ind = []
  yes_ind = []
  s_list = []
  
  # Feature Values
  
  x = deepcopy(dataset3[:, :-1])
  
  # Target Values
  
  y = deepcopy(dataset3[:, -1])
  
  for c, i in enumerate(y):
    if (i == 0):
      no_ind.append(c)
    else:
      yes_ind.append(c)

  # Normalizing
  
  m1 = np.mean(x[:,0])
  m2 = np.mean(x[:,1])
  sd1 = stdev(x[:,0])
  sd2 = stdev(x[:,1])
  x[:,0] = (x[:,0] - m1)/sd1
  x[:,1] = (x[:,1] - m2)/sd2

  # Adding x0

  x = np.c_[np.ones((x.shape[0], 1)), x]

  # Initialise Theeta Parameter to Zero

  theta = np.zeros((3, 1))
  lr = 0.01
  epochs = 1000
  cost_p_e = []

  for e in range(epochs):
    cost = 0
    for i in range(100):
      temp = theta

      # Calculate Theta_Transpose*X to pass to sigmoid function

      par = np.dot([x[i]], theta)

      # Return Sigmoid Value and Pass to gradient function

      s = sigmoid(par)

      # Saving Prediction Values for accuracy calculation

      if e == (epochs - 1):
        s_list.append(s)

      cost = cost + cost_func(s, np.array([x[i]]), y[i])
      temp = theta - (lr * gradient(s, np.array([x[i]]), y[i]).T)
      theta = temp
    
    # Append cost of each epoch

    cost_p_e.append(cost/100)

  # Denormalize Data

  x1 = (x[:,1] * sd1) + m1
  x2 = (x[:,2] * sd2) + m2
  
  x_values = [np.min(x[:,1]), np.max(x[:,2])]
  y_values =  -(theta[0] + np.dot(theta[1], np.array([x_values]))) /theta[2]
  
  # Denormalize Decision Boundary
  x_values = (np.array(x_values) * sd1) + m1
  y_values = (y_values * sd2) + m2
 
  print("Part (B) Parameters \n\n", theta, "\n")
  print("\nPart (C)")

  plt.figure()
  plt.plot(np.arange(1,epochs+1,1), cost_p_e)
  plt.ylabel("Cost Function")
  plt.xlabel("Epochs")
  plt.title("SGD")
  plt.show()

  # Calculating Accuracy
  
  print('Part (D)', '\033[91m' + '\033[1m' + 'Accuracy = ' + '\033[0m',
        '\033[1m' + str(accuracy(s_list, y)) + '%' + '\033[0m' + '\n' )
  
  print('Part (E & F)')
  plt.figure()
  plt.plot(x_values,y_values, label = "Decision Boundary")
  plt.scatter(x1[yes_ind], x2[yes_ind], label='One')
  plt.scatter(x1[no_ind], x2[no_ind], label='Zero')
  plt.ylabel("Feature 2")
  plt.xlabel("Feature 1")
  plt.title("Logistic Regression")
  plt.legend(fancybox=True, framealpha=0.5)
  plt.show()
  
SGD(dataset3)

