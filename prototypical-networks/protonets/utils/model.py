import torch 
from tqdm import tqdm

from protonets.utils import filter_opt
from protonets.models import get_model

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from scripts.predict.few_shot.tsne import plot_3dtsne, plot_2dtsne
import seaborn as sns
from sklearn import preprocessing
import itertools
import matplotlib
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)

def evaluate(model, data_loader, meters, desc=None):
    model.eval()

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    
    le = preprocessing.LabelEncoder()
    targets = []
    x = []
    class_prototypes= {}
    prototype_targets = []    

    for sample in data_loader:

        label = (sample['class'])
        labels = list(itertools.chain.from_iterable(itertools.repeat(x, 5) for x in label))
        #labels = labels_support
        #labels.extend(labels_support)
        #print(labels)
        targets.extend(labels)
        #prototype_targets.extend(label)
 

        embeddings, class_prototype, _, output = model.loss(sample)

        embeddings = embeddings.cpu().detach().numpy()
        class_prototype = class_prototype.cpu().detach().numpy()
        #print(class_prototype.shape)

        # prototype_targets.extend(label)
        # for (i,j) in enumerate(label):
        #     if j not in class_prototypes.keys():
        #         class_prototypes[j] = class_prototype[i]
        #     else:
        #         np.append(class_prototypes[j],class_prototype[i])

        x.extend(embeddings)

        
        for field, meter in meters.items():
            meter.add(output[field])

    y = le.fit_transform(targets)
    x = torch.FloatTensor(x)
    class_names = set(targets) 
    class_numbers = set(y)

   

    # pr = []
    # for k in class_prototypes.keys():
    #     pr.append(class_prototypes[k])

    # pr = torch.FloatTensor(pr)
    # print(pr.shape)


    

    prototargets = le.fit_transform(list(class_prototypes.keys()))
    print(prototargets)

    plot_2dtsne(x, y, pr, prototargets, class_names, class_numbers)

    plot_3dtsne(x,y, pr, prototargets, class_names, class_numbers)

    return meters


    
def inference(model, data_loader, meters, desc=None):
    
    print("Inside Inference block")
    model.eval()

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    distances_all = []
    z_all = []
    images = []
    for sample in data_loader:
        #print(sample['image'])
        images.extend(sample['image'])
    
        z, distances = model.loss(sample, inference=True)

        #print(distances.shape)

        z = z.cpu().detach().numpy()
        distances = distances.cpu().detach().numpy()

        distances_all.extend(distances)
    
        z_all.extend(z)
     
    
    
    z_all = torch.FloatTensor(z_all)
    distances_all = torch.FloatTensor(distances_all)

    print(distances_all.shape)
    print(z_all.shape)


    ### Kmeans

    # kmeans = KMeans(n_clusters=100, random_state=1234, algorithm='full', max_iter=800)
    # clusters = kmeans.fit_predict(z_all)
    # print(clusters.shape)
    # print(kmeans.cluster_centers_.shape)
    # #labels=np.array([clusters.labels_])

    # for (i,j) in zip(images,clusters ):
    #     print('%s : %d' % (i,j), file=open("/MLStamps/few-shot/infer.txt", "a"))
    
   

    ###KNN
    model = AgglomerativeClustering(n_clusters=100)
    yhat = model.fit_predict(z_all)
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
	# get row indexes for samples with this cluste
        row_ix = where(yhat == cluster)
        print(row_ix)
	# create scatter of these samples
       



    



