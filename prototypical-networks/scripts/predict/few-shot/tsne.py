import matplotlib.patheffects as PathEffects

import seaborn as sns
import torch
from sklearn.manifold import TSNE
import time

import matplotlib.pyplot as plt
import numpy as np
time_start = time.time()
means = torch.load('results/best_model.pt')
means.eval()
meansnp = means.detach()
fashion_tsne = TSNE().fit_transform(meansnp)
cl = [i for i in range(means.size()[0])]
cl = np.array(cl)
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=7)

        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig('foo.png')

    return f, ax, sc, txts


f, ax, sc, txts = fashion_scatter(fashion_tsne,cl)
print(type(txts[0]))
print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
