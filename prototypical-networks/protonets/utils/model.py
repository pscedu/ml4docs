from tqdm import tqdm
import torch
from protonets.utils import filter_opt
from protonets.models import get_model
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    for sample in data_loader:
   
        z, Y , _ , output = model.loss(sample)

        with torch.no_grad():
            current_outputs = z.cpu().numpy()
            current_labels = Y.cpu().numpy()
            labels = np.concatenate ((Y, current_labels))
            features = np.concatenate((z, current_outputs))

        
        

        for field, meter in meters.items():
            meter.add(output[field])
            #print(output[field])

    print(features.shape)
    print(labels.shape)
    tsne = TSNE(n_components=2).fit_transform(features)

    #print(tsne.shape)

    tx = tsne[:, 0]
    ty = tsne[:, 1]
    plt.scatter(tx, ty, c=labels, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig("tsne.png")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for label in colors_per_class:
    #     indices = [i for i, l in enumerate(labels) if l == label]
    #     current_tx = np.take(tx, indices)
    #     current_ty = np.take(ty, indices)
    #     color = np.array(colors_per_class[label], dtype=np.float) / 255
    #     ax.scatter(current_tx, current_ty, c=color, label=label)
        
    # ax.legend(loc='best')
    # plt.savefig("tsne.png")




    return meters
