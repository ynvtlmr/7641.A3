# coding: utf-8

# In[1]:

import os
import timeit
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import RandomForestClassifier

from helpers import get_abspath
import warnings

warnings.filterwarnings('ignore')


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def dict_to_csv(dict_data, csv_file):
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)


# In[2]:


train_df = pd.read_csv('../data/optdigits_train.csv', header=None)
digits_y = train_df.iloc[:, -1:].as_matrix().flatten()
digits_X = train_df.iloc[:, :-1].as_matrix()

train_df = pd.read_csv('../data/abalone_train.csv', header=None)
abalone_y = train_df.iloc[:, -1:].as_matrix().flatten()
abalone_X = train_df.iloc[:, :-1].as_matrix()

# In[3]:


km = KMeans(random_state=0)  # K-Means
gmm = GMM(random_state=0)  # Gaussian Mixture Model (EM)
km.set_params(n_clusters=3)
gmm.set_params(n_components=11)
km.fit(abalone_X)
gmm.fit(abalone_X)
cols = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight",
        "Male", "Female", "Infant"]

df_gmm = pd.DataFrame(columns=cols, data=abalone_X)
df_km = pd.DataFrame(columns=cols, data=abalone_X)
df_gmm['cluster'] = gmm.predict(abalone_X)
df_km['cluster'] = km.predict(abalone_X)

# In[4]:


df = df_gmm.groupby('cluster')['Male', 'Female', 'Infant'].agg(['sum'])
df.columns = ["Male", "Female", "Infant"]
print(df.T)

# In[5]:


df = df_km.groupby('cluster')['Male', 'Female', 'Infant'].agg(['sum'])
df.columns = ["Male", "Female", "Infant"]
print(df.T)

# In[6]:


test_df = pd.read_csv('../data/optdigits_test.csv', header=None)
digits_y_test = test_df.iloc[:, -1:].as_matrix().flatten()
digits_X_test = test_df.iloc[:, :-1].as_matrix()

test_df = pd.read_csv('../data/abalone_test.csv', header=None)
abalone_y_test = test_df.iloc[:, -1:].as_matrix().flatten()
abalone_X_test = test_df.iloc[:, :-1].as_matrix()

# In[7]:


digits_scalar = StandardScaler()

digits_scalar.fit(digits_X)
digits_X = digits_scalar.transform(digits_X)
digits_X_test = digits_scalar.transform(digits_X_test)

# In[8]:


abalone_scalar = StandardScaler()

abalone_scalar.fit(abalone_X)
abalone_X = abalone_scalar.transform(abalone_X)
abalone_X_test = abalone_scalar.transform(abalone_X_test)

# In[9]:


## BASELINE
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)

start_time = timeit.default_timer()
ann.fit(digits_X, digits_y)
predict = ann.predict(digits_X_test)
x = classification_report(predict, digits_y_test)
print(x)
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# In[10]:


km = KMeans(random_state=0)  # K-Means
gmm = GMM(random_state=0)  # Gaussian Mixture Model (EM)
km.set_params(n_clusters=10)
gmm.set_params(n_components=8)
km.fit(digits_X)
gmm.fit(digits_X)
digits_X_km = np.concatenate(
    (digits_X, get_one_hot(km.predict(digits_X), 10)),
    axis=1
)
digits_X_gmm = np.concatenate(
    (digits_X, get_one_hot(gmm.predict(digits_X), 8)),
    axis=1
)
digits_X_test_km = np.concatenate(
    (digits_X_test, get_one_hot(km.predict(digits_X_test), 10)),
    axis=1
)
digits_X_test_gmm = np.concatenate(
    (digits_X_test, get_one_hot(gmm.predict(digits_X_test), 8)),
    axis=1
)

# In[11]:


##KMEANS
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_gmm, digits_y)
predict = ann.predict(digits_X_test_gmm)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# In[12]:


##GMM
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_km, digits_y)
predict = ann.predict(digits_X_test_km)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# # PCA

# In[13]:


pca = PCA(random_state=0, svd_solver='full', n_components=32)
pca.fit(digits_X)
digits_X_PCA = pca.transform(digits_X)
digits_X_test_PCA = pca.transform(digits_X_test)

# In[14]:


ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_PCA, digits_y)
predict = ann.predict(digits_X_test_PCA)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# In[15]:


km = KMeans(random_state=0)  # K-Means
gmm = GMM(random_state=0)  # Gaussian Mixture Model (EM)
km.set_params(n_clusters=20)
gmm.set_params(n_components=12)
km.fit(digits_X_PCA)
gmm.fit(digits_X_PCA)
digits_X_PCA_km = np.concatenate(
    (digits_X_PCA, get_one_hot(km.predict(digits_X_PCA), 20)),
    axis=1
)
digits_X_PCA_gmm = np.concatenate(
    (digits_X_PCA, get_one_hot(gmm.predict(digits_X_PCA), 12)),
    axis=1
)
digits_X_test_PCA_km = np.concatenate(
    (digits_X_test_PCA, get_one_hot(km.predict(digits_X_test_PCA), 20)),
    axis=1
)
digits_X_test_PCA_gmm = np.concatenate(
    (digits_X_test_PCA, get_one_hot(gmm.predict(digits_X_test_PCA), 12)),
    axis=1
)

# In[16]:


## PCA + KMEANS
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_PCA_km, digits_y)
predict = ann.predict(digits_X_test_PCA_km)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# In[17]:


## PCA + GMM
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()

ann.fit(digits_X_PCA_gmm, digits_y)

predict = ann.predict(digits_X_test_PCA_gmm)

print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# ## ICA

# In[18]:


ica = FastICA(random_state=0, max_iter=5000, tol=1e-04, n_components=22)
ica.fit(digits_X)
digits_X_ICA = ica.transform(digits_X)
digits_X_test_ICA = ica.transform(digits_X_test)

# In[19]:


ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_ICA, digits_y)
predict = ann.predict(digits_X_test_ICA)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# In[20]:


km = KMeans(random_state=0)  # K-Means
gmm = GMM(random_state=0)  # Gaussian Mixture Model (EM)
km.set_params(n_clusters=20)
gmm.set_params(n_components=12)
km.fit(digits_X_ICA)
gmm.fit(digits_X_ICA)

digits_X_ICA_km = np.concatenate(
    (digits_X_ICA, get_one_hot(km.predict(digits_X_ICA), 20)),
    axis=1
)
digits_X_ICA_gmm = np.concatenate(
    (digits_X_ICA, get_one_hot(gmm.predict(digits_X_ICA), 12)),
    axis=1
)

digits_X_test_ICA_km = np.concatenate(
    (digits_X_test_ICA, get_one_hot(km.predict(digits_X_test_ICA), 20)),
    axis=1
)
digits_X_test_ICA_gmm = np.concatenate(
    (digits_X_test_ICA, get_one_hot(gmm.predict(digits_X_test_ICA), 12)),
    axis=1
)

# In[21]:


## ICA + KMEANS
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_ICA_km, digits_y)
predict = ann.predict(digits_X_test_ICA_km)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# In[22]:


## ICA + GMM
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_ICA_gmm, digits_y)
predict = ann.predict(digits_X_test_ICA_gmm)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# ## RP

# In[23]:


rp = SparseRandomProjection(random_state=0, n_components=29)

rp.fit(digits_X)

digits_X_RP = rp.transform(digits_X)
digits_X_test_RP = rp.transform(digits_X_test)

# In[24]:


ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_RP, digits_y)
predict = ann.predict(digits_X_test_RP)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# In[25]:


km = KMeans(random_state=0)  # K-Means
gmm = GMM(random_state=0)  # Gaussian Mixture Model (EM)
km.set_params(n_clusters=3)
gmm.set_params(n_components=9)
km.fit(digits_X_RP)
gmm.fit(digits_X_RP)

digits_X_RP_km = np.concatenate(
    (digits_X_RP, get_one_hot(km.predict(digits_X_RP), 3)),
    axis=1
)
digits_X_RP_gmm = np.concatenate(
    (digits_X_RP, get_one_hot(gmm.predict(digits_X_RP), 9)),
    axis=1
)

digits_X_test_RP_km = np.concatenate(
    (digits_X_test_RP, get_one_hot(km.predict(digits_X_test_RP), 3)),
    axis=1
)
digits_X_test_RP_gmm = np.concatenate(
    (digits_X_test_RP, get_one_hot(gmm.predict(digits_X_test_RP), 9)),
    axis=1
)

# In[26]:


## RP + KMEANS
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_RP_km, digits_y)
predict = ann.predict(digits_X_test_RP_km)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# In[27]:


## RP + GMM
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_RP_gmm, digits_y)
predict = ann.predict(digits_X_test_RP_gmm)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# ## Random Forest

# In[28]:


rfc = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=0)
fi = rfc.fit(digits_X, digits_y).feature_importances_
i = [i + 1 for i in range(len(fi))]
fi = pd.DataFrame({'importance': fi, 'feature': i})
fi.sort_values('importance', ascending=False, inplace=True)
fi['i'] = i
cumfi = fi['importance'].cumsum()

idxs = fi.loc[:cumfi.where(cumfi > 0.8).idxmin(), :]
idxs = list(idxs.index)

digits_X_RF = digits_X[:, idxs]
digits_X_test_RF = digits_X_test[:, idxs]

# In[29]:


ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_RF, digits_y)
predict = ann.predict(digits_X_test_RF)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# In[30]:


km = KMeans(random_state=0)  # K-Means
gmm = GMM(random_state=0)  # Gaussian Mixture Model (EM)
km.set_params(n_clusters=11)
gmm.set_params(n_components=9)
km.fit(digits_X_RF)
gmm.fit(digits_X_RF)

digits_X_RF_km = np.concatenate(
    (digits_X_RF, get_one_hot(km.predict(digits_X_RF), 11)),
    axis=1
)
digits_X_RF_gmm = np.concatenate(
    (digits_X_RF, get_one_hot(gmm.predict(digits_X_RF), 9)),
    axis=1
)

digits_X_test_RF_km = np.concatenate(
    (digits_X_test_RF, get_one_hot(km.predict(digits_X_test_RF), 11)),
    axis=1
)
digits_X_test_RF_gmm = np.concatenate(
    (digits_X_test_RF, get_one_hot(gmm.predict(digits_X_test_RF), 9)),
    axis=1
)

# In[31]:


## RP + KMEANS
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_RF_km, digits_y)
predict = ann.predict(digits_X_test_RF_km)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))

# In[32]:


## RF + GMM
ann = MLPClassifier(
    activation='relu', max_iter=20,
    solver='adam', learning_rate='adaptive',
    hidden_layer_sizes=(20), alpha=0.01
)
start_time = timeit.default_timer()
ann.fit(digits_X_RF_gmm, digits_y)
predict = ann.predict(digits_X_test_RF_gmm)
print(classification_report(predict, digits_y_test))
end_time = timeit.default_timer()
elapsed = end_time - start_time
print("Done: {} seconds".format(elapsed))
