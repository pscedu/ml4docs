"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import pylab as pl
from sklearn import svm
import time

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            train_dataset = facenet.get_dataset(args.train_data_dir)
            test_dataset = facenet.get_dataset(args.test_data_dir)
                 
            paths_train, labels_train = facenet.get_image_paths_and_labels(train_dataset)
            paths_test, labels_test = facenet.get_image_paths_and_labels(test_dataset)
            
            print('Number of train classes: %d' % len(train_dataset))
            print('Number of train images: %d' % len(paths_train))
            
            print('Number of test classes: %d' % len(test_dataset))
            print('Number of test images: %d' % len(paths_test))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for TRAIN images')
            nrof_images = len(paths_train)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array_train = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths_train[start_index:end_index]
                train_images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:train_images, phase_train_placeholder:False }
                emb_array_train[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            print('Calculating features for TEST images')
            nrof_images = len(paths_test)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array_test = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths_test[start_index:end_index]
                test_images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:test_images, phase_train_placeholder:False }
                emb_array_test[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)


            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                class_names = [ cls.name.replace('__', ' ') for cls in train_dataset]

                print(class_names)
                print(len(class_names))
                #model = SVC(kernel='linear', probability=True)
                #model = SVC(kernel='rbf' ,probability=True,gamma=0.7,C=1.0)

                #model.fit(emb_array, labels)
                
                #kfold=KFold(n_splits=5, shuffle=True, random_state=0)
                #cv_scores=cross_val_score(model, emb_array, labels, cv=kfold)
                #print("Cross Validation score for SVC Linear: ", cv_scores)
       
                #C = 1.0

                print(emb_array_train.shape)
                feat_cols_train = [ 'pixel'+str(i) for i in range(emb_array_train.shape[1]) ]
                df_train = pd.DataFrame(emb_array_train,columns=feat_cols_train)
                df_train['y'] = labels_train
                df_train['label'] = df_train['y'].apply(lambda i: str(class_names[i]))
                X_train, y_train = None, None
                print('Size of the dataframe: {}'.format(df_train.shape))
          
                print(emb_array_test.shape)
                feat_cols_test = [ 'pixel'+str(i) for i in range(emb_array_test.shape[1]) ]
                df_test = pd.DataFrame(emb_array_test,columns=feat_cols_test)
                df_test['y'] = labels_test
                df_test['label'] = df_test['y'].apply(lambda i: str(class_names[i]))
                X_test, y_test = None, None
                print('Size of the dataframe: {}'.format(df_test.shape))



                #pca = PCA(n_components=3)
                #pca_result = pca.fit_transform(df[feat_cols].values)
                #df['pca-one'] = pca_result[:,0]
                #df['pca-two'] = pca_result[:,1] 
                #df['pca-three'] = pca_result[:,2]
                #print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

                df_subset_train = df_train.copy()
                data_subset_train = df_subset_train[feat_cols_train].values

                df_subset_test = df_test.copy()
                data_subset_test = df_subset_test[feat_cols_test].values

                time_start = time.time()
                tsne = TSNE(n_components=2, verbose=0, perplexity=5, learning_rate= 100.0, n_iter=500)

                tsne_results_train = tsne.fit_transform(data_subset_train)
                tsne_results_test = tsne.fit_transform(data_subset_test)

                print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

                df_subset_train['tsne-2d-one'] = tsne_results_train[:,0]
                df_subset_train['tsne-2d-two'] = tsne_results_train[:,1]

                df_subset_test['tsne-2d-one'] = tsne_results_test[:,0]
                df_subset_test['tsne-2d-two'] = tsne_results_test[:,1] 

                plt.figure(figsize=(16,10))

                concatenated = pd.concat([df_subset_train.assign(dataset='df_subset_train'), df_subset_test.assign(dataset='df_subset_test')])

                sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", data=concatenated, hue='label', style='dataset', legend="full", s=100,alpha=0.8)

                #sns.scatterplot( x="tsne-2d-one", y="tsne-2d-two", hue="label",palette=sns.color_palette("husl", 29), data=df_subset_train,legend="full",s=60,alpha=0.8, marker='o')
                #sns.scatterplot( x="tsne-2d-one", y="tsne-2d-two", hue="label",palette=sns.color_palette("husl", 29), data=df_subset_test,s=100, legend=False, alpha=0.8, marker='+')
                plt.savefig("tsne.png")
                #plt.figure(figsize=(16,7))

                #ax3 = plt.subplot(1, 3, 3)
                #`sns.scatterplot( x="tsne-pca50-one", y="tsne-pca50-two", hue="y", palette=sns.color_palette("hls", 10), data=df_subset,legend="full", alpha=0.3,ax=ax3)

                #X= pca_result[:, :2]
                #svc = svm.SVC(kernel='linear', C=C).fit(X, labels)
                #rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, labels)
                #poly_svc = svm.SVC(kernel='poly', probability=True, degree=3, C=C).fit(X, labels)
                #lin_svc = svm.LinearSVC(C=C).fit(X, labels)

                #h = .02  # step size in the mesh
                
                # create a mesh to plot in
                #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                #xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

                # title for the plots
                #l = [class_names[labels[i]] for i in range(len(labels))]
                #cmap = pl.cm.Paired
                # extract all colors from the .jet map
                #cmaplist = [cmap(i) for i in range(cmap.N)]
                 # create the new map
                #cmap = pl.cm.from_list('Custom cmap', cmaplist, cmap.N)
                #bounds = np.linspace(0,29,29+1)
                #norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                #bounds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
                #titles = ['SVC with linear kernel', 'SVC with RBF kernel', 'SVC with polynomial (degree 3) kernel','LinearSVC (linear kernel)']
                #for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
                    #Plot the decision boundary. For that, we will assign a color to each
                    #point in the mesh [x_min, m_max]x[y_min, y_max].
                #    pl.subplot(2, 2, i + 1)
                #    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

                    # Put the result into a color plot
                #    Z = Z.reshape(xx.shape)
                #    pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
                #    pl.axis('off')

                    # Plot also the training points
                #    pl.scatter(X[:, 0], X[:, 1], c=labels,cmap=pl.cm.Paired )
                    #sns.scatterplot( x="test", y="test", hue="label",palette=sns.color_palette("husl", 29), data=df_subset,legend="full",s=60,alpha=0.8)
#plt.legend(scat, l ,scatterpoints=1,loc='lower left',ncol=3,fontsize=8)
                    #pl.legend()

                #    pl.title(titles[i])

                #cb = pl.colorbar(spacing='proportional',ticks=bounds)
                #cb.set_ticklabels(class_names)
                #plt.legend()
                #pl.show() 
                #pl.savefig('embeddings.png')     

                # Create a list of class names
                #class_names = [ cls.name.replace('__', ' ') for cls in data_subset_train]                
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('train_data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('test_data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_false')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=100)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=0)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
