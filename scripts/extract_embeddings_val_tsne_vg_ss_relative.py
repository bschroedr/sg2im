#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pdb
import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random
import numpy as np
import numpy.matlib
import pdb
import pprint
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
#from tsne import bh_sne
#import pandas as pd
#import seaborn as sns
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, squareform
from sg2im.heatmapcluster import heatmapcluster

from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator, CondGANPatchDiscriminator, CondGANDiscriminator
#from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard

# 59% model
from sg2im.model_layout_ss import Sg2ImModel
#from sg2im.model_layout_vg import Sg2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager
import sg2im.db_utils as db_utils

####
#from sg2im.perceptual_loss import FeatureExtractor
from sg2im.data.utils import imagenet_deprocess_batch
#from sg2im.data.utils import perc_process_batch
#from sg2im.logger import Logger

from imageio import imwrite
import sg2im.vis as vis

#torch.backends.cudnn.benchmark = True

VG_DIR = os.path.expanduser('/home/brigit/sandbox/sg2im_brigit/datasets/vg')
#VG_DIR = os.path.expanduser('/home/brigit/sandbox/sg2im_brigit/datasets/vg')
#VG_DIR = os.path.expanduser('/home/brigit/sandbox/sg2im/datasets/vg')
COCO_DIR = os.path.expanduser('/home/brigit/datasets/coco_stuff')
USER_DIR = os.path.expanduser('/home/brigit/')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=100000, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

# COCO-specific options
parser.add_argument('--coco_train_image_dir',
         default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--coco_val_image_dir',
         default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--coco_train_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple) # supercategories
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)
parser.add_argument('--random_seed', default=42, type=int)  #For comparative results/debug, use in val_loader
# data augmentation
parser.add_argument('--heuristics_ordering', default=True, type=bool_flag)
# extreme points (this assumes you have extreme points!
parser.add_argument('--coco_instances_extreme_train_json',
         #default=os.path.join(COCO_DIR, 'annotations/instances_extreme_train2017.json'))
         default=None)
parser.add_argument('--coco_instances_extreme_val_json',
         #default=os.path.join(COCO_DIR, 'annotations/instances_extreme_val2017.json'))
         default=None)

# triplet options
parser.add_argument('--triplet_box_net', default=False, type=int)
parser.add_argument('--triplet_superbox_net', default=False, type=int)
parser.add_argument('--triplet_mask_size', default=0, type=int)
# embedding for contextual search
parser.add_argument('--triplet_embedding_size', default=0, type=int)
# predict additioal information  for bbox (e.g. 4 pts + 'meta' info)
# use object masks as prior for triplet
parser.add_argument('--masks_to_triplet_mlp', default=False, type=int)
parser.add_argument('--masks_to_triplet_pixels', default=False, type=int)
# use masked SG loss
parser.add_argument('--use_masked_sg', default=False, type=int)
# triplet context (output) dimension
parser.add_argument('--triplet_context_size', default=384, type=int)
# show retrieval visualization
parser.add_argument('--visualize_retrieval', default=False, type=int)
# show oracle visualization
parser.add_argument('--visualize_oracle', default=False, type=int)
# use input embeddings for evaluation 
parser.add_argument('--raw_features', default=False, type=int)
parser.add_argument('--min_iou', default=0.2, type=float)
# relative spatial relevance
parser.add_argument('--relative_iou', default=False, type=int)
# visualize relative vs absolute spatial relevance
parser.add_argument('--compare_iou', default=False, type=int)
parser.add_argument('--generate_samples', default=False, type=int)
# retrieval type
parser.add_argument('--lvrr', default=False, type=int)
parser.add_argument('--vrr', default=False, type=int)
parser.add_argument('--ssr', default=False, type=int)
 
# load object embeddings db from file
parser.add_argument('--coco_object_db_json', default=None, type=str)
# filename to save object embeddings db to
parser.add_argument('--coco_object_db_json_write', default=None, type=str)
# label for model for saving images
parser.add_argument('--model_label', default='test', type=str)
# whitelist for triplet subject classes
parser.add_argument('--triplet_obj_whitelist', default=None, type=str)
# load triplet embeddings db from file
parser.add_argument('--coco_triplet_db_json', default=None, type=str)
# filter by object type /per 
parser.add_argument('--obj_class', default=None, type=str)

# Generator options
parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default='1024,512,256,128,64', type=int_tuple)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

## scene_graph conditioning on GAN
parser.add_argument('--sg_context_dim', default=0, type=int)
parser.add_argument('--sg_context_dim_d', default=0, type=int)
parser.add_argument('--image_patch_discr', default=True, type=bool_flag)
parser.add_argument('--gcnn_pooling', default='sum')
parser.add_argument('--layout_for_discrim', default=False, type=bool_flag)


parser.add_argument('--matching_aware_loss', default=False, type=bool_flag)

# Generator losses
parser.add_argument('--mask_loss_weight', default=0, type=float)
parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
parser.add_argument('--predicate_pred_loss_weight', default=0, type=float) # DEPRECATED
parser.add_argument('--perceptual_loss_weight', default=0, type=float)
parser.add_argument('--grayscale_perceptual', action='store_true', help='Calculate perceptual loss with grayscale images')
parser.add_argument('--log_perceptual', action='store_true', help='Take logarithm of perceptual loss')

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
parser.add_argument('--gan_loss_type', default='gan')
parser.add_argument('--d_clip', default=None, type=float)
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')

# Object discriminator
parser.add_argument('--d_obj_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=32, type=int)
parser.add_argument('--d_obj_weight', default=1.0, type=float) # multiplied by d_loss_weight
parser.add_argument('--ac_loss_weight', default=0.1, type=float)

# Image discriminator
parser.add_argument('--d_img_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--d_img_weight', default=1.0, type=float) # multiplied by d_loss_weight

# Output options
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--output_dir', default='./test/')

parser.add_argument('--checkpoint', default='sg2im-models/coco64.pt')
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])



def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1, logarithm=False):
  curr_loss = curr_loss * weight
  if logarithm:
    curr_loss = torch.log(curr_loss)
  loss_dict[loss_name] = curr_loss.item()
  if total_loss is not None:
    total_loss += curr_loss
  else:
    total_loss = curr_loss
  return total_loss

def check_args(args):
  H, W = args.image_size
  for _ in args.refinement_network_dims[1:]:
    H = H // 2
  if H == 0:
    raise ValueError("Too many layers in refinement network")


def build_model(args, vocab):
  kwargs = {
    'vocab': vocab,
    'image_size': args.image_size,
    'embedding_dim': args.embedding_dim,
    'gconv_dim': args.gconv_dim,
    'gconv_hidden_dim': args.gconv_hidden_dim,
    'gconv_num_layers': args.gconv_num_layers,
    'mlp_normalization': args.mlp_normalization,
    'refinement_dims': args.refinement_network_dims,
    'normalization': args.normalization,
    'activation': args.activation,
    'mask_size': args.mask_size,
    'layout_noise_dim': args.layout_noise_dim,
    'triplet_box_net': args.triplet_box_net,
    'triplet_mask_size': args.triplet_mask_size,
    'triplet_embedding_size': args.triplet_embedding_size,
    'use_bbox_info': args.use_bbox_info,
    'triplet_superbox_net': args.triplet_superbox_net,
    'masks_to_triplet_mlp': args.masks_to_triplet_mlp,
    'masks_to_triplet_pixels': args.masks_to_triplet_pixels,
    'use_masked_sg': args.use_masked_sg,
    'triplet_context_size': args.triplet_context_size,
  }
  model = Sg2ImModel(**kwargs)
  return model, kwargs


def build_coco_dsets(args):
  dset_kwargs = {
    'image_dir': args.coco_train_image_dir,
    'instances_json': args.coco_train_instances_json,
    'stuff_json': args.coco_train_stuff_json,
    'extreme_pts_json': args.coco_instances_extreme_train_json,
    'stuff_only': args.coco_stuff_only,
    'image_size': args.image_size,
    'mask_size': args.mask_size,
    'max_samples': args.num_train_samples,
    'min_object_size': args.min_object_size,
    'min_objects_per_image': args.min_objects_per_image,
    'instance_whitelist': args.instance_whitelist,
    'stuff_whitelist': args.stuff_whitelist,
    'include_other': args.coco_include_other,
    'include_relationships': args.include_relationships,
    'seed': args.random_seed, # truly random set to 0   
    'heuristics_ordering' : args.heuristics_ordering
  }

  train_dset = None
  #train_dset = CocoSceneGraphDataset(**dset_kwargs)
  #num_objs = train_dset.total_objects()
  #num_imgs = len(train_dset)
  #print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
  #print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

  dset_kwargs['image_dir'] = args.coco_val_image_dir
  dset_kwargs['instances_json'] = args.coco_val_instances_json
  dset_kwargs['stuff_json'] = args.coco_val_stuff_json
  # extreme points
  dset_kwargs['extreme_pts_json'] = args.coco_instances_extreme_val_json
  dset_kwargs['max_samples'] = args.num_val_samples
  #  *deactivate* randomization for val (for consistent results)
  dset_kwargs['seed'] = args.random_seed
  val_dset = CocoSceneGraphDataset(**dset_kwargs)
  num_objs = val_dset.total_objects()
  num_imgs = len(val_dset)
  print('Validation dataset has %d images and %d objects' % (num_imgs, num_objs))
  #print('(%.2f objects per image)' % (float(num_objs) / num_imgs))
 
  #assert train_dset.vocab == val_dset.vocab
  #vocab = json.loads(json.dumps(train_dset.vocab))
  vocab = json.loads(json.dumps(val_dset.vocab))

  return vocab, train_dset, val_dset

def build_vg_dsets(args):
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
    'seed': args.random_seed,
  }
  #train_dset = VgSceneGraphDataset(**dset_kwargs)
  #iter_per_epoch = len(train_dset) // args.batch_size
  #print('There are %d iterations per epoch' % iter_per_epoch)
  train_dset = None

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = VgSceneGraphDataset(**dset_kwargs)

  return vocab, train_dset, val_dset

def build_loaders(args):
  if args.dataset == 'vg':
    vocab, train_dset, val_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn
  elif args.dataset == 'coco':
    vocab, train_dset, val_dset = build_coco_dsets(args)
    collate_fn = coco_collate_fn

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': True,
    'collate_fn': collate_fn,
  }
  #train_loader = DataLoader(train_dset, **loader_kwargs)
  train_loader = None

  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)
  return vocab, train_loader, val_loader


#def check_model(args, t, loader, model, logger=None, log_tag='', write_images=False):
def check_model(args, t, loader, model, log_tag='', write_images=False):

  if torch.cuda.is_available():
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
  else:
    float_dtype = torch.FloatTensor
    long_dtype = torch.LongTensor

  skip_obj_db = False
  ## if specified load saved triplet embeddings
  if args.coco_triplet_db_json is not None:
     print('Loading triplet database: ', args.coco_triplet_db_json)
     triplet_db = db_utils.read_fr_JSON(args.coco_triplet_db_json)
     # if triplet object whitelist, strip out keys without this object
     skip_obj_db = True # don't process anything in triplet/object processing loops; don't run model
  else:
     triplet_db = None
  ## if specified load saved object embeddings
  if args.coco_object_db_json is not None:
     object_db = db_utils.read_fr_JSON(args.coco_object_db_json)
     skip_obj_db = True # let's be explicit
  elif skip_obj_db == False: 
    ## begin extract embedding data from model
    num_samples = 0
    all_losses = defaultdict(list)
    total_iou = 0
    total_boxes = 0
    t = 0
    # relationship (triplet) database
    triplet_db = dict()
    object_db = dict()
    img_id = 0

    # iterate over all batches of images
    with torch.no_grad():
      for batch in loader:
  
        if torch.cuda.is_available():
          batch = [tensor.cuda() for tensor in batch]
        else:
          batch = [tensor for tensor in batch]

        masks = None
        if len(batch) == 8: # VG
          imgs, objs, boxes, triples, obj_to_img, triple_to_img, num_attrs, attrs = batch
          triplet_masks = None
        elif len(batch) == 11: # COCO
          imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, triplet_masks, extreme_points, cat_words, cat_ids = batch
        predicates = triples[:, 1]
 
        # Run the model as it has been run during training
        model_masks = masks
        model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks, tr_to_img=triple_to_img)
        ####model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks,)
        #imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
        imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triple_boxes_pred, triple_boxes_gt, triplet_masks_pred, boxes_pred_info, triplet_superboxes_pred, obj_scores, pred_mask_gt, pred_mask_scores, context_tr_vecs, input_tr_vecs, obj_class_scores, rel_class_scores, subj_scores, rel_embedding, mask_rel_embedding, pred_ground = model_out # mapc_bind = model_out
        # Run model without GT boxes to get predicted layout masks
        #model_out = model(objs, triples, obj_to_img)
        #layout_boxes, layout_masks = model_out[5], model_out[6]
      
        num_batch_samples = imgs.size(0)
        num_samples += num_batch_samples
        if num_samples >= args.num_val_samples:
          break

        # deprocess (normalize) images
        imgs = imgs.detach()
        # deprocess done in data loader
        #samples = {}
        #samples['gt_imgs'] = imgs
        #for k, v in samples.items():
        #  samples[k] = imagenet_deprocess_batch(v)
        np_imgs = [gt.cpu().numpy().transpose(1,2,0) for gt in imgs]
         
        super_boxes = []
 
        # use fine-tuned RELATION EMBEDDING 
        pred_embeddings = rel_embedding 

        num_obj_class = len(model.vocab['object_idx_to_name'])
        num_pred_class = len(model.vocab['pred_idx_to_name'])

        ### embedding stuff below here ####
        for i in range(0,num_batch_samples):
          #print('Processing image', i+1, 'of batch size', args.batch_size)
          #f.write('---------- image ' + str(i) + '----------\n')

          # process objects and get embeddings, per image 
          objs_index = np.where(obj_to_img.cpu() == i)[0] # objects indices for image in batch 
          objs_img = objs[objs_index] # object class id labels for image
          #obj_names = np.array(model.vocab['object_idx_to_name'])[objs_img] # index with object class index
          obj_names = np.array(model.vocab['object_idx_to_name'])[objs_img.cpu().numpy()] 

          # object supercategories : dictionary with string cat ids as keys, supercategory strings as value
          #obj_supercats = []
          #count = 0
          #for id in objs_img.cpu().numpy():
          #  if id == 0:
          #    sc = 'other'
          #  else:
          #    str_id = str(id)
          #    sc = model.vocab['object_idx_to_supercat'][str_id]
            #print(id, obj_names[count], sc)
          #  obj_supercats += [sc]
          #  count += 1

          # scene graph embedding
          obj_embeddings_img = obj_embeddings[objs_index]
          # word embedding
          #obj_embeddings_word = cat_words[objs_index]
          num_objs = len(objs_index)
       
          for j in range(0, num_objs):
            name = obj_names[j]

            # object whitelist: select only these objects for DB

            entry = {'id': objs_img[j].tolist(), 'embed': obj_embeddings_img[j].tolist()}
            if name not in object_db:
              object_db[name] = dict() 
              object_db[name]['objs'] = [entry]
              object_db[name]['count'] = 1
              #object_db[name]['word_embed'] = obj_embeddings_word[j].tolist() # word embedding
            elif name in object_db:
              object_db[name]['objs'] += [entry]
              object_db[name]['count'] += 1
            #XXXX 
            #f.write('obj ' + str(i) + ': ' + name + '\n')  

          #continue
          # test if db is serializable
          #pdb.set_trace() 
          #import json
          #jd = json.dumps(object_db)

	  # process all triples for in image
          tr_index = np.where(triple_to_img.cpu().numpy() == i)
          tr_img = triples[tr_index].cpu()
          pred_embeddings_img = pred_embeddings[tr_index]
          # batch index
          batch_index = np.unique(triple_to_img[tr_index].cpu()).item()
	  # 8 point triple boxes
          np_triple_boxes_gt = np.array(triple_boxes_gt.cpu()).astype(float)
          tr_img_boxes = np_triple_boxes_gt[tr_index]
          assert len(tr_img) == len(tr_img_boxes)

	  # triplet context: acts like encoded "signature" for each SG
          #tr_context = context_tr_vecs[tr_index]
	  # input embeddings 
          tr_input = input_tr_vecs[tr_index]

          # TEST 
	  # MAPC binding
          #tr_mapc = mapc_bind[tr_index]

	  # vocab['object_idx_to_name'], vocab['pred_idx_to_name']
	  # s,o: indices for "objs" array (yields 'object_idx' for 'object_idx_to_name')
	  # p: use this value as is (yields 'pred_idx' for 'pred_idx_to_name')
          s, p, o = np.split(tr_img, 3, axis=1)

	  # iterate over all triplets in image to form (subject, predicat, object) tuples
          relationship_data = []
          num_triples = len(tr_img)
          # predicted triplet boxes
          pred_tr_boxes = np.concatenate((np.array(boxes_pred[s]), np.array(boxes_pred[o])), axis=1)
          pred_tr_boxes = np.reshape(pred_tr_boxes, (num_triples, 8))

	  # need to iterate over all triples due to information that needs to be extracted per triple
          for n in range(0, num_triples):
	    # tuple = (objs[obj_index], p, objs[subj_index])
            subj_index = s[n]
            subj = np.array(model.vocab['object_idx_to_name'])[objs[subj_index]]
            # pred supercat
            #cat_id = objs[subj_index].cpu().numpy().item()
            #if cat_id == 0:
            #  subj_supercat = 'other'
            #else:
            #  subj_supercat = model.vocab['object_idx_to_supercat'][str(cat_id)]

            pred = np.array(model.vocab['pred_idx_to_name'])[p[n]]
            obj_index = o[n]
            obj = np.array(model.vocab['object_idx_to_name'])[objs[obj_index]]
            # triple one-hot vector
            subj_onehot = np.zeros(num_obj_class)
            pred_onehot = np.zeros(num_pred_class)
            obj_onehot = np.zeros(num_obj_class)
            subj_onehot[objs[subj_index]] = 1 
            pred_onehot[p[n]] = 1
            obj_onehot[objs[obj_index]] = 1 
            vrr_onehot = np.concatenate((subj_onehot,pred_onehot,obj_onehot), axis=0)
            ssr_onehot = np.concatenate((subj_onehot,obj_onehot), axis=0)

	    # pred supercat
            #cat_id = objs[obj_index].cpu().numpy().item()
            #if cat_id == 0:
            #  obj_supercat = 'other'
            #else:
            #  obj_supercat = model.vocab['object_idx_to_supercat'][str(cat_id)]

            # object whitelist: select only these triplets for triplet DB
            if args.triplet_obj_whitelist is not None and subj != args.triplet_obj_whitelist and obj != args.triplet_obj_whitelist: 
              continue 
            else:
              if args.triplet_obj_whitelist is not None:
                print('>>> processing object class: ', args.triplet_obj_whitelist)

            triplet = tuple([subj, pred, obj]) 
            relationship_data += [triplet]
            #print(tuple([subj, pred, obj]))
            #print(tr_img[n])
            #print('--------------------')
            #f.write('(' + db_utils.tuple_to_string(tuple([subj, pred, obj])) + ')\n')
           
            # discard singletons 
            if pred == '__in_image__':
              continue

            # GT bounding boxes: (x0, y0, x1, y1) format, in a [0, 1] coordinate system
	    # (from "boxes" (one for each object in "objs") using subj_index and obj_index)
            subj_bbox = tr_img_boxes[n,0:4] 
            obj_bbox = tr_img_boxes[n, 4:8] 
            pred_subj_bbox = pred_tr_boxes[n,0:4]
            pred_obj_bbox = pred_tr_boxes[n,4:8]
	    #print(tuple([subj, pred, obj]), subj_bbox, obj_bbox)

	    # SG GCNN embeddings
            subj_embed = obj_embeddings[subj_index].cpu().numpy().tolist()
            pred_embed = pred_embeddings_img[n].cpu().numpy().tolist()
            obj_embed = obj_embeddings[obj_index].cpu().numpy().tolist()
            #pooled_embed = (obj_embeddings[subj_index].cpu().numpy() + pred_embeddings[n].cpu().numpy() + obj_embeddings[obj_index].cpu().numpy())/3
            aa = obj_embeddings[subj_index].cpu().numpy().squeeze()
            bb = pred_embeddings_img[n].cpu().numpy() 
            cc = obj_embeddings[obj_index].cpu().numpy().squeeze()
            pooled_embed = np.concatenate((aa, bb, cc)).tolist()
            #context_embed = tr_context[n].cpu().numpy()
            #context_embed = np.concatenate((context_embed, aa, bb, cc)).tolist()
            #mapc_embed = tr_mapc[n].cpu().numpy()
            input_embed = tr_input[n].cpu().numpy()
            inout_embed = np.concatenate((input_embed, aa, bb, cc)).tolist()
            lvrr_raw_features_embed = np.concatenate((vrr_onehot, subj_bbox, obj_bbox), axis=0) 
            ssr_raw_features_embed = np.concatenate((ssr_onehot, subj_bbox, obj_bbox), axis=0) 
            
            relationship = dict()
            relationship['subject'] = subj
            relationship['predicate'] = pred
            relationship['object'] = obj
            #relationship['subject_supercat'] = subj_supercat
            #relationship['object_supercat'] = obj_supercat
            relationship['subject_bbox'] = subj_bbox.tolist() #JSON can't serialize np.array()
            relationship['object_bbox'] = obj_bbox.tolist()
            relationship['subject_name'] = subj 
            relationship['object_name'] = obj 
            
	    # get super box
            min_x = np.min([subj_bbox[0], obj_bbox[0]])
            min_y = np.min([subj_bbox[1], obj_bbox[1]])
            max_x = np.max([subj_bbox[2], obj_bbox[2]])
            max_y = np.max([subj_bbox[3], obj_bbox[3]])
            relationship['super_bbox'] = [min_x, min_y, max_x, max_y]
            min_x = np.min([pred_subj_bbox[0], pred_obj_bbox[0]])
            min_y = np.min([pred_subj_bbox[1], pred_obj_bbox[1]])
            max_x = np.max([pred_subj_bbox[2], pred_obj_bbox[2]])
            max_y = np.max([pred_subj_bbox[3], pred_obj_bbox[3]])
            relationship['pred_super_bbox'] = [min_x, min_y, max_x, max_y]
            super_boxes += [relationship['super_bbox']]
            relationship['subject_embed'] = subj_embed
            relationship['predicate_embed'] = pred_embed
            relationship['object_embed'] = obj_embed
            relationship['embed'] = pooled_embed
            # contextual embedding
            #relationship['context_embed'] = context_embed
            #relationship['mapc_embed'] = mapc_embed
            relationship['input_embed'] = input_embed
            relationship['raw_vrr'] = vrr_onehot 
            relationship['raw_ssr'] = ssr_raw_features_embed 
            relationship['raw_lvrr'] = lvrr_raw_features_embed 
            # assign triplet an image id
            relationship['img_id'] = img_id

            # image patch
            H, W = args.image_size
            x0, x1 = int(round(min_x*W)), int(round(max_x*W))
            y0, y1 = int(round(min_y*H)), int(round(max_y*H))
            # print('batch index = ', batch_index)
            #print(relationship['super_bbox'])
            ###pdb.set_trace()
            #patch = np_imgs[batch_index][y0:y1 + 1, x0:x1+1]
            #plt.imshow(patch); plt.show()
            # np arrays are not serializable
            # to recover np array: patch = np.array(relationship['image'])
            #relationship['image'] = [patch.tolist()] 
            #relationship['image'] = patch 
            #sb_overlay_image = vis.overlay_boxes(np_imgs[batch_index], model.vocab, objs_vec, 
            #                   torch.from_numpy(np.array([relationship['super_bbox']])), obj_to_img, W=args.image_size[0], H=args.image_size[1], 
            #                   drawText=False, drawSuperbox=True)
            sb_overlay_image = vis.overlay_boxes(np_imgs[batch_index], model.vocab, objs_vec, 
                               torch.from_numpy(np.array([relationship['subject_bbox'], 
                               relationship['object_bbox']])), obj_to_img, 
                               W=args.image_size[0], H=args.image_size[1], 
                               drawText=False, drawSuperbox=False)
            relationship['full_image'] = [sb_overlay_image]
            # create user study image 
            white_image = np.ones((args.image_size[0], args.image_size[1], 3))
            bbox_overlay_image = vis.overlay_boxes(white_image, model.vocab, objs_vec, 
                               torch.from_numpy(np.array([relationship['subject_bbox']])), obj_to_img, 
                               W=args.image_size[0], H=args.image_size[1], 
                               drawText=False, drawSuperbox=False)
            relationship['white_image'] = [bbox_overlay_image]
            #relationship['full_image'] = [np_imgs[batch_index].tolist()] 

            triplet_str = db_utils.tuple_to_string(triplet)
            if triplet_str not in triplet_db:
              triplet_db[triplet_str] = [relationship]
            elif triplet_str in triplet_db:
              triplet_db[triplet_str] += [relationship]
          
          # increment image id
          img_id += 1

          #print('------- end of processing for image --------------------------')

        #####f.close()
        # measure IoU as a basic metric for bbox prediction
        total_iou += jaccard(boxes_pred, boxes)
        total_boxes += boxes_pred.size(0)
        ####### end single batch process #########
        print('------- single batch processing --------------------------')
        #pdb.set_trace()
     
    #f.close()
    # write object database to JSON file
    #pdb.set_trace()
    if args.coco_object_db_json is None and args.coco_object_db_json_write is not None: # if no db specified to load
      print('Writing database to file: ', args.coco_object_db_json_write)
      db_utils.write_to_JSON(object_db, args.coco_object_db_json_write)
    # write triplet database to JSON file
    if args.coco_triplet_db_json is None and triplet_db is not None: 
      if args.triplet_obj_whitelist is not None:
        triplet_label = args.triplet_obj_whitelist + '.json' # tree, person
      else: 
        triplet_label = 'coco_triplet_db.json'
      #db_utils.write_to_JSON(triplet_db, triplet_label) 
  #####  end embedding extraction

  # analyze JSON database (stats, examples,etc)
  # also, compare word vs SG embedding
  if triplet_db is not None:
    analyze_embedding_retrieval(triplet_db)
  
  out = []

  ####################
  avg_iou = total_iou / total_boxes
  print('average bbox IoU = ', avg_iou.cpu().numpy())
  ###################

  return tuple(out)

# calculate recall
def calculate_recall(results, count):
  num_elem = len(results)
  all_r = [] 
  for i in range(0,num_elem):
    r = sum(results[:i+1])/count # normal results 
    #r = min(i+1,count)/count # ideal results
    all_r += [r]
  return np.array(all_r)

def calculate_IoU(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
  # compute the area of intersection rectangle
  #interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
  interArea = max(0, xB - xA) * max(0, yB - yA)
  # compute the area of both the prediction and ground-truth rectangles
  #boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  #boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
  # compute the intersection over union by taking the intersection area
  # and dividing it by the sum of prediction + ground-truth areas (interesection area)
  IoU = interArea / float(boxAArea + boxBArea - interArea)
  return IoU

def calculate_relative_IoU(q_sbox, r_sbox, q_bbox, r_bbox):
  # crop superboxes
  # bbox = [x0 y0 x1 y1]
  w_qsb = q_sbox[2] - q_sbox[0] 
  h_qsb = q_sbox[3] - q_sbox[1]
  w_rsb = r_sbox[2] - r_sbox[0] 
  h_rsb = r_sbox[3] - r_sbox[1]
  #print("w/h query sbox, w/h retr sbox: ", w_qsb, h_qsb, w_rsb, h_rsb)
  # get rescale factor from superboxes to apply to q_bbox, r_bbox
  w_scale = w_qsb/w_rsb
  h_scale = h_qsb/h_rsb
  #print("w/h scale: ",w_scale, h_scale)
  # crop bounding boxes
  w_rbb = r_bbox[2] - r_bbox[0]
  h_rbb = r_bbox[3] - r_bbox[1]
  #print("w/h retrieval bbox: ", w_rbb, h_rbb)
  # rescale r_bbox to calculate IoU between it and q_bbox
  w_rbb_sc = w_rbb*w_scale
  h_rbb_sc = h_rbb*h_scale
  #print("scaled w/h of retrieval bbox: ", w_rbb_sc, h_rbb_sc)
  # rescale r_bbox translate vector in query box frame scale  and translate into query box frame
  x1_rbb_sc = (r_bbox[0]-r_sbox[0])*w_scale + q_sbox[0]
  y1_rbb_sc = (r_bbox[1]-r_sbox[1])*h_scale + q_sbox[1]
  #print(r_bbox[0], r_sbox[0])
  #print(r_bbox[1], r_sbox[1])
  x2_rbb_sc = x1_rbb_sc + w_rbb_sc
  y2_rbb_sc = y1_rbb_sc + h_rbb_sc
  #print([x1_rbb_sc, y1_rbb_sc, x2_rbb_sc, y2_rbb_sc])
  #print(q_bbox)
  # calculate IoU between q_bbox and rescaled r_bbox
  rel_IoU = calculate_IoU(q_bbox, [x1_rbb_sc, y1_rbb_sc, x2_rbb_sc, y2_rbb_sc]) 
  return rel_IoU 

# this places the box in a normalized coordinate system. 
# subject and object box will be compared once in (shared) normalized coord system.
def relativize(box,parent):
  w = parent[2] - parent[0]
  h = parent[3] - parent[1]
  return [
          (box[0] - parent[0]) / w,
          (box[1] - parent[1]) / h,
          (box[2] - parent[0]) / w,
          (box[3] - parent[1]) / h,
  ]

###
def euclid_dist(t1, t2):
  t = (t1-t2)
  tt = t**2
  tts = tt.sum(axis=1)
  ttsr = np.sqrt(tts)
  return ttsr
  #return np.sqrt(((t1-t2)**2).sum(axis = 1))

def analyze_embedding_retrieval(db):
  # fix retrieval order
  #random.seed(1)

  hist = {}
  for key in db.keys():
    l = len(db[key])
    # look for non unique triplets
  #  if(l > 1):
  #    print('found triplet with >1 entry:', key)
  #    pdb.set_trace()
    #print(key,':', l)
    #hist[key] = l
    hist[key] = {'count':l}

  # sort by value
  #sort_hist = sorted(hist, key=lambda x: hist[x], reverse=True) # x is key
  sort_triplets_by_count = sorted(hist, key=lambda x: hist[x].get('count',-1), reverse=True)
  sort_count = [hist[k]['count'] for k in sort_triplets_by_count]

  # plot distribution of sorted triplets
  #plt.clf() # clear current axes
  #stc = sort_triplets_by_count[0:200]
  #sc = sort_count[0:200]
  #x_pos = np.arange(len(sc))
  #plt.bar(x_pos, sc, align='center', alpha=0.5)
  #plt.xticks(x_pos, stc)
  #plt.yticks(np.arange(0, max(sc)+1, 2.0))
  #plt.set_xlim((0,200))
  #plt.ylabel("triplet count")
  #plt.xlabel("triplet label")
  #plt.title("MS-COCO: Distribution of Top 200 Triplets")
  #plt.setp(plt.gca().get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=6)
  #plt.tight_layout()
  #plt.show()

  x = []
  embeds = []
  imgs = []
  white_imgs = []
  np_imgs = []
  labels = []
  labels_to_keys = []
  subjects = []
  objects = []
  img_ids = []
  iou = []
  su_bbox = []
  ob_bbox = []
  superbox = []

  # randomize triplet index
  su = []
  pr = []
  ob = []

  if(len(sort_triplets_by_count) == 0):
    print('ERROR: is triplet database empty?')
    pdb.set_error()

  # iterate over all triplets in db
  for k in sort_triplets_by_count:
    tr = db_utils.string_to_tuple(k)
    subj = tr[0] # subject
    obj = tr[2] # object

    # iterate over list of triplets
    for t in range(0, len(db[k])):
      # report results using this
      if not args.raw_features: 
          embeds += [db[k][t]['embed']]
      else:
        if args.vrr == True:
          embeds += [db[k][t]['raw_vrr']] 
        elif args.lvrr == True:
          embeds += [db[k][t]['raw_lvrr']] 
        elif args.ssr == True:
          embeds += [db[k][t]['raw_ssr']] 
      
      su += db[k][t]['subject_embed'] 
      ob += db[k][t]['object_embed'] 
      pr += [ db[k][t]['predicate_embed'] ]

      # calculate IoU
      su_bbox += [db[k][t]['subject_bbox']]
      ob_bbox += [db[k][t]['object_bbox']]
      superbox += [db[k][t]['super_bbox']]


      # image patch
      #imgs += [ db[k][t]['image'] ]
      # full image
      imgs += [ db[k][t]['full_image'] ]
      white_imgs += [ db[k][t]['white_image'] ]

      # L-VRR (spatial + semantic)/VRR (semantic only)
      if args.lvrr == True or args.vrr == True:
        labels += [k]
      # SSR (subj/object only + spatial)
      elif args.ssr == True:
        labels += [str(db[k][t]['subject_name'] + ' ' + db[k][t]['object_name'])]
      ##labels += [ k + '\n[' + db[k][t]['subject_supercat'] + '||' + db[k][t]['object_supercat'] + ']' ]
      labels_to_keys += [k]
      subjects += [subj]
      objects += [obj]
      img_ids += [db[k][t]['img_id']]

  # randomize predicate
  #triplet_idx = np.random.permutation(len(pr))
  #pr = np.array(pr)[triplet_idx].tolist()
  #embeds = np.concatenate((su, pr, ob), axis=1).tolist()

  # gives all keys in database
  #keys = [k for k in db]
  np_keys = np.array(labels)
  # sort keys alphabetically
  #sort_keys = sorted(keys, reverse = True)
  
  # select x queries 
  #query_idx = np.random.permutation(np.arange(len(embeds)))
  # how many total triplets are in top 200 (by label)
  # NOOOOOOOOO!!!!!!!!!!!!!
  num_keys = len(np_keys)
  sc = np.array(sort_count[:num_keys])
  #sc = np.array(sort_count)
  # !!!!!!!!!!!!!!!!!!!!!!1

  #print('set up query index')
  # set up query index
  # use end of long tail distribution
  use_longtail = False  
  if use_longtail == False:
    # large-scale distribution (per JJ's SG retrieval paper)
    count_idx = np.where(sc > 1) #np.sum(sc[count_idx])
    #count_idx = sc[np.where(sc > 6)] #np.sum(sc[count_idx]) # jj
    count = np.sum(sc[count_idx])
    start_idx = 0
    end_idx = count-1
  else:
    # end of long-tail distribution
    count_idx = np.where((sc < 4) & (sc > 1))
    count = np.sum(sc[count_idx])
    # start/end range
    one_count = np.sum(sc[np.where(sc < 2)])
    start_idx = np.sum(sc[np.where(sc > 3)]) 
    end_idx = len(embeds) - one_count - 1
    count = np.sum(sc[count_idx])
  # build query index
  query_idx = np.arange(start_idx, end_idx+1)
  # large-scale distribution
  #query_idx = np.random.permutation(np.arange(count)) 
  ##### randomize
  np.random.seed(args.random_seed) # use numpy random seed!!
  query_idx = np.random.permutation(query_idx)

  #print(np_keys[:num_keys])
  visualize_exs = True
  print('number of unfiltered triplets:', len(embeds))
  f = open('all_queries_docs.txt', 'w')
  if args.generate_samples:
    f_s = open(args.output_dir + '/image_iou_xywh_list.txt', 'w')

  if(visualize_exs):
    t = 0
    topK_recall = 100 
    topK = 10 
    total_recall = []
    recall = []
    find_recall = True
    mean_iou = []
    # THOUGHT: should we be retrieving patches from the same scene graph/image?
    # objects in this will/could share embeddings, eg. [person] right of floor, [person] under table
    # QUESTION: are embedding from the same SG clustered near each other???
    # scene graph centroid? capture context with this - use as another search embedding!! <s/p/o/context>
    for i in query_idx:
      k = np.copy(np_keys)
      # make local copy of embeds, make query value very large
      tmp_embeds = np.copy(embeds)
      tmp_embeds[i] = 999 
      # find top K matches
      # filter by subject, such as person
      subj = subjects[i]
      obj = objects[i]
      if args.obj_class is not None:
        if subj != args.obj_class and obj != args.obj_class: 
          continue
      query = embeds[i]
      query_str = np_keys[i]
      query_img = imgs[i]
      white_query_img = white_imgs[i]
      query_orig_idx = i
      # IoU
      query_su_bbox = su_bbox[i]
      query_ob_bbox = ob_bbox[i]
      query_superbox = superbox[i]

      k[i] = "[query]\n" + k[i]
      # query image id
      query_img_id = img_ids[i]
      dist = euclid_dist(np.asarray(query), np.asarray(tmp_embeds))
      index = dist.argsort(axis=0) # indices of sorted list

      # find matching img ids to remove query img from ranked results
      id_idx = np.where(np.array(img_ids) == query_img_id)
      img_triplets = np.array(labels)[id_idx]
      tr_count = len(np.where(img_triplets== query_str)[0]) # includes query triplet
      dist[id_idx] = 999.99
      # mark all labels as query for these triplets
      k[id_idx] = "[query]\n" + query_str
 
      # sort retrieved distances with those of query type "omitted"
      index = dist.argsort(axis=0)
      
      # write all queries/docs to file 
      #f.write(query_str + '\n') 

      # calculate recall
      if(find_recall):
        # Random result
        #index = np.random.permutation(index) # randomize sorted index

        # semantic relevance
        results = k[index] # index is sorted index
        #results = k[index[0:topK_recall]]
        #qq = np.matlib.repmat(query_str,topK_recall,1).squeeze()
        qq = np.matlib.repmat(query_str,len(results),1).squeeze()
        rr = (results == qq).astype(int)

        #IoU
        results_idx = index
        #results_idx = index[0:topK_recall]
        s_iou = []
        o_iou = []
        sb_iou = []
        results_iou = []
        results_iou_bool = []
        rel_iou_bool = [] 
        abs_iou_bool = [] 
        rel_iou = [] 
        abs_iou = [] 
        rr_all = []
        min_iou = args.min_iou
        #min_superbox_iou = 0.5
        min_superbox_iou = 0.4 # decrease SB thresh, incr su/ob thresh

        for b in results_idx:
          # superbox
          retr_superbox = superbox[b]
          # subject/object bbox
          retr_su_bbox = su_bbox[b]
          retr_ob_bbox = ob_bbox[b]
          # triplet superbox IoU for relative IoU calculation
          superbox_iou = calculate_IoU(query_superbox, retr_superbox) 
          su_iou = 0
          ob_iou = 0
          if args.relative_iou: 
            if superbox_iou > min_superbox_iou:
              su_iou = calculate_relative_IoU(query_superbox, retr_superbox, query_su_bbox, retr_su_bbox) 
              ob_iou = calculate_relative_IoU(query_superbox, retr_superbox, query_ob_bbox, retr_ob_bbox) 
              # alternative method that gives same result
              #q_relative = relativize(query_su_bbox,query_superbox)
              #r_relative = relativize(retr_su_bbox,retr_superbox)
              #su_iou = calculate_IoU(q_relative, r_relative)
              #q_relative = relativize(query_ob_bbox,query_superbox)
              #r_relative = relativize(retr_ob_bbox,retr_superbox)
              #ob_iou = calculate_IoU(q_relative, r_relative)
            results_iou_bool += [(su_iou >= min_iou) and (ob_iou >= min_iou)]
          else:
            su_iou = calculate_IoU(query_su_bbox, retr_su_bbox)
            ob_iou = calculate_IoU(query_ob_bbox, retr_ob_bbox)
            results_iou_bool += [(su_iou >= min_iou) and (ob_iou >= min_iou)] 

          #results_iou_bool += [(su_iou >= min_iou) and (ob_iou >= min_iou)] 
          s_iou += [su_iou]
          o_iou += [ob_iou]
          sb_iou += [superbox_iou]
  
          # we know index of image 
          if args.compare_iou:
            #su_iou_abs = ob_iou_abs = 0
            #su_iou_rel = ob_iou_rel = 0
            #if superbox_iou > min_superbox_iou:
            su_iou_rel = calculate_relative_IoU(query_superbox, retr_superbox, query_su_bbox, retr_su_bbox)
            ob_iou_rel = calculate_relative_IoU(query_superbox, retr_superbox, query_ob_bbox, retr_ob_bbox)
            su_iou_abs = calculate_IoU(query_su_bbox, retr_su_bbox)
            ob_iou_abs = calculate_IoU(query_ob_bbox, retr_ob_bbox)
            rel_iou_bool += [(su_iou_rel >= min_iou) and (ob_iou_rel >= min_iou) and (superbox_iou > min_superbox_iou)]
            abs_iou_bool += [(su_iou_abs >= min_iou) and (ob_iou_abs >= min_iou)]
            rel_iou += [(superbox_iou, su_iou_rel, ob_iou_rel)]
            abs_iou += [(su_iou_abs, ob_iou_abs)]
 
        ##rr_iou = (np.array(results_iou) >= min_iou).astype(int)
        rr_iou = (np.array(results_iou_bool) == True).astype(int)
        # consider spatial + semantic relevance for L-VRR or SSR
        if args.lvrr == True or args.ssr == True:
          rr_all  = np.logical_and(rr, rr_iou).astype(int)   
        # consider semantic relevance only for VRR:
        elif args.vrr == True:
          rr_all  = rr
        # ideal: flip ones and zeros, sort.
        #rr_not = np.logical_not(rr_all).astype(int) 
        #rr_index = rr_not.argsort(axis=0)
        #rr_all = rr_all[rr_index]

        match_total = np.sum(rr_all)
        #mean_iou += [results_iou]
        mean_iou += [[np.sum(rr_iou)]]
        #pdb.set_trace()

        if args.vrr == True:
          triplet_count = hist[labels_to_keys[i]]['count'] - tr_count # remove triplets that are from same image (as query)
        elif args.lvrr == True or args.ssr == True:
          triplet_count = sum(rr_all)

        min_triplet_count = 1 # 3 - 95% recall
        # calculate topK_recalls for 1 query 
        if(triplet_count >= min_triplet_count): 
        ##if(triplet_count-1) > min_triplet_count:
          recall = calculate_recall(rr_all[0:topK_recall], triplet_count)
          #recall = calculate_recall(rr_all, triplet_count)
          total_recall.append(recall)
          # write query to file 
          f.write(query_str + '(query)\n') 
        #print('query idx = ', i)
      #  skip query that doesn't have at least 1 retrieval result 
        elif(triplet_count < min_triplet_count): 
          print('skipping recall calculation for ', query_str)
          f.write(query_str + '\n') 
          continue
        if match_total == 0:
          print('....SKIPPING....')
          continue

      if args.compare_iou:
        compare_iou_bool = np.logical_xor(rel_iou_bool, abs_iou_bool).astype(int)
        iou_idx = np.where(compare_iou_bool == True) # index into results_idx
        iou_idx = iou_idx[0][0:15]
        query_img = np.array(imgs[query_orig_idx]).squeeze()
        for i in iou_idx:
          fig = plt.figure(figsize=(6,3))
          ax = fig.add_subplot(1,3,1)
          # query img
          plt.imshow(query_img) 
          plt.xlabel(k[query_orig_idx], fontsize=6)
          ax.tick_params(length=0, width=0, labelbottom=False, labelleft=False)
          # retrieved img
          img_index = results_idx[i] # index into (original) imgs list
          retr_img = np.array(imgs[img_index]).squeeze()
          ax = fig.add_subplot(1,3,2)
          plt.imshow(retr_img) # relative
          iou_label = "{:.2f} {:.2f} {:.2f}".format(rel_iou[i][0], rel_iou[i][1], rel_iou[i][2]) # relative
          if rel_iou_bool[i] == True:
            iou_label += ": MATCH"
          plt.xlabel(iou_label, fontsize=6)
          ax.tick_params(length=0, width=0, labelbottom=False, labelleft=False)
          ax = fig.add_subplot(1,3,3)
          plt.imshow(retr_img) # absolute
          iou_label = "{:.2f} {:.2f}".format(abs_iou[i][0], abs_iou[i][1]) # absolute 
          if abs_iou_bool[i] == True:
            iou_label += ": MATCH"
          plt.xlabel(iou_label, fontsize=6)
          ax.tick_params(length=0, width=0, labelbottom=False, labelleft=False)
          img_name = args.output_dir + '/' + args.model_label + '_compare_IoU' + str(query_orig_idx) + '_' + str(i) + '.png'
          plt.savefig(img_name, bbox_inches='tight', pad_inches = 0)
          #plt.show()
        continue

      if args.generate_samples:
        white_query_img = np.array(white_imgs[query_orig_idx]).squeeze()
        idx = index[0:5*topK].tolist()
        for c, n in enumerate(idx): # c is new results (like iou) index, n is index corresponding retr. sorted order (db items)
          iou_label = "{:.2f}".format(s_iou[c])
          fig = plt.figure(figsize=(6,3))
          ax = fig.add_subplot(1,2,1)
          plt.imshow(white_query_img)
          ax.tick_params(length=0, width=0, labelbottom=False, labelleft=False)
          ax = fig.add_subplot(1,2,2)
          retr_img = np.array(white_imgs[n]).squeeze()
          plt.imshow(retr_img)        
          ax.tick_params(length=0, width=0, labelbottom=False, labelleft=False)
          #plt.xlabel(iou_label, fontsize=6)
          img_name = args.output_dir + '/' + str(query_orig_idx) + '_' + str(n) + '_' + iou_label + '.png'
          print('Writing file ', img_name)
          #plt.savefig(img_name, bbox_inches='tight', pad_inches = 0)
          # xywh for query and subject boxes
          w = su_bbox[n][2] - su_bbox[n][0] # use sorted 
          h = su_bbox[n][3] - su_bbox[n][1]  
          xywh = str(su_bbox[n][0]) + '\t' + str(su_bbox[n][1]) + '\t' + str(w) + '\t' + str(h)
          w = query_su_bbox[2] - query_su_bbox[0] 
          h = query_su_bbox[3] - query_su_bbox[1]
          query_xywh = str(query_su_bbox[0]) + '\t' + str(query_su_bbox[1]) + '\t' + str(w) + '\t' + str(h)
          f_s.write(img_name + '\t' +  iou_label + '\t' + query_xywh + '\t' + xywh + '\n')
          #plt.show()
        continue
      
      # skip retrieval visualization for this query and
      # continue on to the next retrieval 
      if not args.visualize_retrieval and not args.visualize_oracle: 
        continue

      # visualize query and topK images
      count = 1
      fig = plt.figure(figsize=(18,3))
      #if args.visualize_oracle:
      #  idx = index[0:100].tolist() # to get all top matches
      #  ax = fig.add_subplot(1,match_total+1, count)
      #else:
      idx = index[0:topK].tolist()
      ax = fig.add_subplot(1,topK+1, count)
      print('QUERY: ', query_str) 
      sm_img = np.array(imgs[query_orig_idx]).squeeze()
      plt.imshow(sm_img)
      ax.tick_params(length=0, width=0, labelbottom=False, labelleft=False)
      plt.xlabel(k[query_orig_idx], fontsize=6)
      count += 1
      for c, n in enumerate(idx):
        iou_label = "{:.2f} {:.2f} {:.2f}".format(sb_iou[c], s_iou[c], o_iou[c])
        x_label = k[n] + '\n' + iou_label
        if args.visualize_oracle:
          if rr_all[c] == True:
            x_label = x_label + '  MATCH'
          #if rr_all[c] == False:
            #continue 
          #ax = fig.add_subplot(1,match_total+1, count)
        #else:
        ax = fig.add_subplot(1,topK+1, count)
        sm_img = np.array(imgs[n]).squeeze()
        plt.imshow(sm_img)
        ax.tick_params(length=0, width=0, labelbottom=False, labelleft=False)
        plt.xlabel(x_label, fontsize=6)
        #plt.xlabel(k[n], fontsize=6)
        print('RESULT:', k[n], s_iou[c], o_iou[c])
        if rr_all[c] == True:
          print('=====MATCH=====')
        count += 1

      if args.visualize_oracle: 
        # this needs to be before plt.show()
        img_name = args.output_dir + '/' + args.model_label + '_IoU' + str(args.min_iou) + '_' + str(query_orig_idx) + '.png'
        plt.savefig(img_name, bbox_inches='tight', pad_inches = 0)
      #pdb.set_trace()
      if args.visualize_retrieval:
        plt.show()
    
      t += 1
 
    # calculate average recall
    if(find_recall):
      if total_recall == []:
        print('no matches found for object class' , args.obj_class)
        pdb.set_trace() 
      mean_recall = np.mean(total_recall, axis=0) # column-wise
      #np.savetxt('mean_recall_topK' + str(topK_recall)+ '_' + args.model_label + '.txt', mean_recall)
      np.savetxt('recall_' + args.model_label + '.txt', np.array(np.mat((mean_recall[0], mean_recall[4], mean_recall[9], mean_recall[99]))), fmt='%10.5f')
      # plot recall
      #fig = plt.figure()
      #plt.style.use('seaborn-whitegrid')
      #plt.ylim(0,1.0)
      #plt.xlim(0, topK_recall)
      #plt.xlabel('k')
      #plt.ylabel('Recall at k')
      #plt.title(args.model_label)
      #x = np.arange(1,topK_recall+1)
      #plt.plot(x, mean_recall)
      #plt.show()
      print('RECALL: r@1 =', mean_recall[0], 'r@5 = ', mean_recall[4], 'r@10 = ', mean_recall[9], 'r@100 = ', mean_recall[99])
      # plot histogram of IoU
      #mean_iou = [item for sublist in mean_iou for item in sublist]
      #fig = plt.figure()
      #plt.hist(mean_iou, bins=100)
      #plt.title('Distribution of IoU')
      #plt.show()
      
    #pdb.set_trace()
    
  f.close()
  f_s.close()

###
def main(args):
  # in order to reproduce results, set seed
  torch.manual_seed(args.random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  if args.lvrr == False and args.vrr == False and args.ssr == False:
    print('Need to specify retrieval type: --lvrr, --vrr, or --ssr')
    exit() 

  if args.device == 'cpu':
    device = torch.device('cpu')
  elif args.device == 'gpu':
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
      print('WARNING: CUDA not available; falling back to CPU')
      device = torch.device('cpu')
  # Load the model, with a bit of care in case there are no GPUs
  map_location = 'cpu' if device == torch.device('cpu') else None
  assert os.path.isfile(args.checkpoint)
  checkpoint = torch.load(args.checkpoint, map_location=map_location)
  # for flags added after model trained.
  checkpoint['model_kwargs']['triplet_box_net'] = args.triplet_box_net
  checkpoint['model_kwargs']['triplet_mask_size'] = args.triplet_mask_size
  checkpoint['model_kwargs']['triplet_embedding_size'] = args.triplet_embedding_size
  checkpoint['model_kwargs']['triplet_context_size'] = args.triplet_context_size
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'], strict=False)
  model.eval()
  model.to(device)

  vocab, train_loader, val_loader = build_loaders(args)
  
  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
    print('Created %s' %args.output_dir)

  if args.visualize_oracle:
    args.visualize_retrieval = True
    args.raw_features = True  
  ## add code for validation visualization
  #logger = Logger(args.output_dir)
  logger = None

  t = 1
  with timeit('forward', args.timing):
    #print('Extracting embeddings from train set:')
    #train_results = check_model(args, t, train_loader, model, log_tag='Train', write_images=False)
    print('Extracting embeddings from val test set:')
    val_results = check_model(args, t, val_loader, model, log_tag='Validation', write_images=False)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
