import spacepy
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from matplotlib import pyplot, dates
from datetime import datetime
import tarfile
import urllib
from kneed import KneeLocator


import os
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
from spacepy import pycdf
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans

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
x_axis = []
data = np.array([])


#Br=np.array([])
#Bt=np.array([])
Bn=np.array([])
data_points=0

for ll in list_files:
    cdf = pycdf.CDF(dir + ll)
    td = cdf['Epoch'][:]
    rad_field = cdf['flowSpeed'][:]
    #rad_bfield=cdf['BR'][:]
    #tan_bfield=cdf['BT'][:]
    nor_bfield=cdf['BN'][:]
    idx = []
    # This eliminates data gaps

    for ii in range(len(rad_field)):
        if (nor_bfield[ii] != -9.999999848243207e+30):
            idx.append(ii) # create array with integer x_axis instead of time variable
            x_axis.append(ii)
            data_points += 1
    y_axis=np.append(Bn,np.array(nor_bfield[idx]))


y_axis_reshape= y_axis.reshape(-1,1)

for i in range(len(x_axis)-1):
    data = np.append(data, y_axis[i])

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

sse = []
for k in range(1, 20): # for 1-20 initial data clusters
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs) # unpack the dictionary into a cluster(k means function)
    kmeans.fit(y_axis_reshape) # standerize
    sse.append(kmeans.inertia_)
plt.style.use("fivethirtyeight") #plotting sse
plt.plot(range(1, 20), sse)
plt.xticks(range(1, 20))
plt.xlabel("Number of Clusters(V)")
plt.ylabel("SSE")
plt.show()
kl = KneeLocator( # algorithim for determining elbow point rather than looking at the graph (in the kneed package)
    range(1, 20), sse, curve="convex", direction="decreasing"
)


print("Bn Cluters: " + str(kl.elbow))


kmeans = KMeans(init="random", n_clusters = kl.elbow, n_init = 10, random_state = 42)
kmeans.fit(y_axis_reshape) # fit to data in scaled features
plt.plot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,0],'o')
fig, ax1 = plt.subplots(sharex=True, sharey=True)
fig.suptitle("Flow Speeds from ", fontsize=16)
fte_colors = {
    0: "#008fd5",
    1: "#fc4f30",
}

kmeans = KMeans(init="random", n_clusters = kl.elbow, n_init = 10, random_state = 42)
kmeans.fit(y_axis_reshape) # fit to data in scaled features
plt.plot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,0],'o')
fig, ax1 = plt.subplots(sharex=True, sharey=True)
fig.suptitle("Flow Speeds from ", fontsize=16)
Bt_list = y_axis_reshape.tolist()
Bt1_list = []
final_data = []
for i in range(len(Bt_list)):
    Bt1_list += Bt_list[i]
    final_data.append([x_axis[i],Bt1_list[i]])
a_data = np.array(final_data)
print(a_data)
ax1.scatter(a_data[:,0], a_data[:,1], c=kmeans.labels_.astype(float))

plt.show()