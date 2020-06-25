import torch 

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import seaborn as sns
import itertools
import matplotlib
import pandas as pd
import numpy as np

def plot_3dtsne(x,y,class_prototypes_values=None,class_proto=None,class_names=None, class_numbers=None): 

    ax = plt.figure(figsize=(16,10)).gca(projection='3d')

    labelTups = zip(list(class_names), list(class_numbers))

    tsne = TSNE(n_components=3)
    q_tsne = tsne.fit_transform(x)
    proto_tsne = tsne.fit_transform(class_prototypes_values)
    
    vis_x_proto = proto_tsne[:, 0]
    vis_y_proto = proto_tsne[:, 1]
    vis_z_proto = proto_tsne[:, 2]

    
    vis_x = q_tsne[:, 0]
    vis_y = q_tsne[:, 1]
    vis_z = q_tsne[:, 2] 

    #data = (vis_x, vis_x, vis_x) 

    sc = ax.scatter(
        vis_x, 
        vis_y,  
        vis_z,
        c=y,
        cmap='tab20'
        )
    # sc = ax.scatter(
    #     vis_x_proto, 
    #     vis_x_proto,  
    #     vis_x_proto,
    #     c=class_proto,
    #     cmap='tab20', marker='x'
    #     )

    colors = [sc.cmap(sc.norm(i)) for i in class_numbers]
    custom_lines = [plt.Line2D([],[], ls="", marker='.', 
                mec='k', mfc=c, mew=.1, ms=20) for c in colors]
    ax.legend(custom_lines, [lt[0] for lt in labelTups], 
          loc='center left', bbox_to_anchor=(0.8, .5))
    
    ax.set_xlabel('tsne-one')
    ax.set_ylabel('tsne-two')
    ax.set_zlabel('tsne-three')
    plt.savefig("foo_test_3d.png")

def plot_2dtsne(x,y, class_prototypes_values,class_proto,class_names, class_numbers):
    #palette = sns.color_palette("bright", len(class_names))
    labelTups = zip(list(class_names), list(class_numbers))

    tsne = TSNE(n_components=2)
    q_tsne = tsne.fit_transform(x)
    proto_tsne = tsne.fit_transform(class_prototypes_values)
    
    vis_x_proto = proto_tsne[:, 0]
    vis_y_proto = proto_tsne[:, 1]
    
    vis_x = q_tsne[:, 0]
    vis_y = q_tsne[:, 1]

    #print(vis_x)
    sc = plt.scatter(vis_x, vis_y, c=y,cmap='tab20')
    #sc = plt.scatter(vis_x_proto, vis_y_proto, c=class_proto,cmap='tab20', marker='x')

    colors = [sc.cmap(sc.norm(i)) for i in class_numbers]
    custom_lines = [plt.Line2D([],[], ls="", marker='.', 
                mec='k', mfc=c, mew=.1, ms=20) for c in colors]
    plt.legend(custom_lines, [lt[0] for lt in labelTups], 
          loc='center left', bbox_to_anchor=(0.8, .5))
    #plt.legend()
    plt.xlabel('tsne-one')
    plt.ylabel('tsne-two')
    
    plt.savefig("foo_test_2d.png")

