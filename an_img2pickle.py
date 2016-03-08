# -*- coding: utf-8 -*-
"""
Read Images and make pickle for Tensorflow
Created on Tue Mar  8 09:17:02 2016

@author: Aneesh
"""

#%% import and define base variables
import glob
import os
import numpy as np
import skimage
from skimage.io import imread_collection
from six.moves import cPickle as pickle


root_root = "/home/nyx/Desktop/Caffe_test/mult_fold_imgs/ZProj"
img_size = 400

#%% Read Dataset

def ab_load_im_fold(fold_name):
    """Load all images from a folder"""
    file_list = glob.glob(os.path.join(fold_name, '*.tif'))
    im_coll = imread_collection(file_list)
    dataset = np.ndarray(shape = (len(im_coll),img_size,img_size),
                         dtype = np.float32)    
    
    for im_idx,im in enumerate(im_coll):        
        dataset[im_idx, :, :] = skimage.img_as_float(im)
        
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def an_pickle(root_root, force=False):
    dataset_names = []
    fold_list = os.listdir(root_root)
    for fold_name in fold_list:
        set_filename = fold_name + '.pickle'
        write_path = os.path.join(root_root,set_filename)
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
          # You may override by setting force=True.
          print('%s already present - Skipping pickling.' % set_filename)
        else:
          print('Pickling %s.' % set_filename)
          dataset = ab_load_im_fold(os.path.join(root_root, fold_name))
          try:
            with open(write_path, 'wb') as f:
              pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
          except Exception as e:
            print('Unable to save data to', set_filename, ':', e)
    return dataset_names

    #
dataset_names = an_pickle(root_root)
   
#%%
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels
#%%
def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(dataset_names)
    valid_dataset, valid_labels = make_arrays(valid_size, img_size)
    train_dataset, train_labels = make_arrays(train_size, img_size)
    
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    
    for label, im_set in enumerate(dataset_names):
        try:
            with open(os.path.join(root_root,im_set), "rb") as f:
                img_set = pickle.load(f)
                np.random.shuffle(img_set)
                if valid_dataset is not None:
                        valid_letter = img_set[:vsize_per_class, :, :]
                        valid_dataset[start_v:end_v, :, :] = valid_letter
                        valid_labels[start_v:end_v] = label
                        start_v += vsize_per_class
                        end_v += vsize_per_class
                print(valid_dataset.shape)
                train_letter = img_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
                
        except Exception as e:
                print('Unable to process data from', im_set, ':', e)
                raise     
    return valid_dataset, valid_labels, train_dataset, train_labels   
    
#%% 
train_size = 700
valid_size = 27
valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  dataset_names, train_size, valid_size)
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
#%%
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
#%%
pickle_file = os.path.join(root_root,'all_cells_set.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,    
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
#%%
valid_labels
#%% Scratch
d=0
for im_set in dataset_names:
    img_set = pickle.load(open(os.path.join(root_root,im_set), "rb"))
    d+=len(img_set)
    print(len(img_set))


d    
                