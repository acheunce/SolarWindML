import spacepy
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot, dates
from datetime import datetime
import tarfile
import urllib
from kneed import KneeLocator


import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
from spacepy import pycdf


from numpy import mean
from numpy import std

def save_elements(list_e, name):
    f = file(name, 'w')
    for el in list_e:
        pickle.dump(el, f)
    f.close()


def read_elements(nb_el, name):
    f = file(name, 'r')
    list_el = []
    for el in range(nb_el):
        list_el.append(pickle.load(f))
    f.close()
    return list_el


def find_outlist(dir, dtype="cdf"):
    outlist = []
    for ff in os.listdir(dir):
        i = len(dtype) + 1
        if ff[-i:] == "." + dtype:
            outlist.append(ff)
    outlist.sort()
    return outlist


dir = "./"
list_files = find_outlist(dir, dtype="cdf")
x_axis = np.array([])
V = np.array([])
#D = np.array([])
T = np.array([])


#Br=np.array([])
#Bt=np.array([])
#Bn=np.array([])
data_points=0

for ll in list_files:
    cdf = pycdf.CDF(dir + ll)
    td = cdf['Epoch'][:]
    velocity = cdf['flowSpeed'][:]
    #density =cdf['protonDensity'][:]
    temperature =cdf['protonTemp'][:]
    #nor_bfield=cdf['BN'][:]
    idx = []
    # This eliminates data gaps

    for ii in range(len(velocity)):
        if (velocity[ii] !=-9.999999848243207e+30 ) and (velocity[ii] != -9.999999848243207e+30): # and (density[ii] != -9.999999848243207e+30)
            idx.append(ii) # create array with integer x_axis instead of time variable
            x_axis = np.append(x_axis, ii)
            data_points += 1
    V = np.append(V, np.array(velocity[idx]))
    T = np.append(T, np.array(temperature[idx]))
    # # calculate summary statistics
    # V_mean, V_std = mean(velocity[idx]), std(velocity[idx])
    # # identify outliers for V
    # cut_offV = V_std * 3
    # lowerV, upperV = V_mean - cut_offV, V_mean + cut_offV
    # outliersV = [x for x in velocity[idx] if x < lowerV or x > upperV]
    # print('Identified Velocity outliers: %d' % len(outliersV))
    # # remove outliers
    # outliersRemovedV = [x for x in velocity[idx] if x >= lowerV and x <= upperV]
    # #print('Non-outlier Velocity observations: %d' % len(outliersRemovedV))


    
    # T_mean, T_std = mean(temperature[idx]), std(temperature[idx])
    # # identify outliers for T
    # cut_offT = T_std * 3
    # lowerT, upperT = T_mean - cut_offT, T_mean + cut_offT
    # outliersT = [x for x in temperature[idx] if x < lowerV or x > upperV]
    # print('Identified Temperature outliers: %d' % len(outliersT))
    # # remove outliers
    # # for x in temperature[idx]
    # #     if x >= lowerT and x <= upperT or if y >= lowerT and y <= upperT:
    # outliersRemovedT = [x for x in temperature[idx] and y for y in velocity[idx] if x >= lowerT and x <= upperT or y >= lowerT and y <= upperT]
    # #print('Non-outlier Temperature observations: %d' % len(outliersRemovedT)) 

#print(V)
#print(T)
finalV = []
for i in range(len(x_axis)):
    finalV.append([V[i], T[i]])
arrV = np.array(finalV)
#print(finalV)
print(arrV[:,0])
print(arrV[:,1])
kmeans_kwargs = {
    "init": "random",
    "n_init": 20,
    "max_iter": 500,
    "random_state": 42,
}

scaler = StandardScaler()
scaled = scaler.fit_transform(arrV)
sseFinalV = []
for k in range(1, 20): # for 1-20 initial data clusters
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs) # unpack the dictionary into a cluster(k means function)
    kmeans.fit(scaled) # standarize
    sseFinalV.append(kmeans.inertia_)
plt.style.use("fivethirtyeight") #plotting sse
plt.plot(range(1, 20), sseFinalV)
plt.xticks(range(1, 20))
plt.xlabel("Number of Clusters(V)")
plt.ylabel("SSE")
#plt.show()
klFinalV = KneeLocator( # algorithim for determining elbow point rather than looking at the graph (in the kneed package)
    range(1, 20), sseFinalV, curve="convex", direction="decreasing"
)






print("Velocity Clusters: " + str(klFinalV.elbow))
# print("Bt Clusters: " + str(kl_Br.elbow))
# print("Br Clusters: " + str(kl_Bt.elbow))
# print("Bn Clusters: " + str(kl_Bn.elbow))

fig, (ax1) = plt.subplots(
    1, 1, figsize=(12, 8),)
fig.suptitle(f"Univariate Clustering Velocity and Temperature", fontsize=16)
fte_colors = {
    0: "#008fd5",
    1: "#fc4f30",
}

# take out of array and rearrange from list, then put into array for (x,y) format

# clustering veloctiy
kmeansV = KMeans(init="random", n_clusters = klFinalV.elbow, n_init = 10, random_state = 42)
kmeansV.fit(scaled) # fit to data in scaled features
#a_data = np.array(arrV)
ax1.scatter(scaled[:,0], scaled[:,1], c=kmeansV.labels_.astype(float))
plt.xlabel("Velocity ", fontdict={"fontsize": 12})
plt.ylabel("Temperature ", fontdict={"fontsize": 12})



plt.show()