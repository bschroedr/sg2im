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
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models


from sg2im.data.coco_ep import CocoSceneGraphDataset, coco_collate_fn
#from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator, CondGANPatchDiscriminator, CondGANDiscriminator
from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard, relation_score

from sg2im.model_layout_ep import Sg2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager

####
from sg2im.perceptual_loss import FeatureExtractor
from sg2im.data.utils import imagenet_deprocess_batch
from sg2im.data.utils import perc_process_batch
from sg2im.logger import Logger


ERASE_LINE_STD = '\x1b[2K'
UP_LINE_STD = '\033[F'


#torch.backends.cudnn.benchmark = True

VG_DIR = os.path.expanduser('/dataset/vg') 
COCO_DIR = os.path.expanduser('/dataset/coco_stuff')


#COCO_DIR = '/public/coco_stuff'


def validate_args(args):
    if not os.path.isdir(args.output_dir):
        print('Creating output directory [%s]' %(args.output_dir))
        os.makedirs(args.output_dir)

    H, W = args.image_size
    for _ in args.refinement_network_dims[1:]:
        H = H // 2
    if H == 0:
        raise ValueError("Too many layers in refinement network")

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp)

    return args


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--kubernetes', default=False, type=bool_flag, help='Using kubernetes')
  parser.add_argument('--dataset', default='coco', choices=['vg', 'coco'], help='Dataset to use')

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
  parser.add_argument('--loader_num_workers', default=16, type=int)
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
  parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
  parser.add_argument('--coco_include_other', default=False, type=bool_flag)
  parser.add_argument('--min_object_size', default=0.02, type=float)
  parser.add_argument('--min_objects_per_image', default=3, type=int)
  parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)
  parser.add_argument('--random_seed', default=42, type=int)  #For comparative results/debug, use in val_loader
  parser.add_argument('--coco_view_validation_error', default=False, type=bool_flag)
  # data augmentation
  parser.add_argument('--heuristics_ordering', default=True, type=bool_flag) 
  # extreme points
  parser.add_argument('--coco_instances_extreme_train_json',
           default=os.path.join(COCO_DIR, 'annotations/instances_extreme_train2017.json'))
  parser.add_argument('--coco_instances_extreme_val_json',
           default=os.path.join(COCO_DIR, 'annotations/instances_extreme_val2017.json'))  

  # triplet options
  parser.add_argument('--triplet_box_net', default=False, type=int)  
  parser.add_argument('--triplet_superbox_net', default=False, type=int)  
  parser.add_argument('--triplet_mask_size', default=0, type=int)  
  # embedding for contextual search
  parser.add_argument('--triplet_embedding_size', default=0, type=int)  
  # predict additioal information  for bbox (e.g. 4 pts + 'meta' info)
  parser.add_argument('--use_bbox_info', default=False, type=int)  
  # use object masks as prior for triplet 
  parser.add_argument('--masks_to_triplet_mlp', default=False, type=int)  
  parser.add_argument('--masks_to_triplet_pixels', default=False, type=int)  

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
  parser.add_argument('--mask_loss_weight', default=0.1, type=float)
  parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
  parser.add_argument('--bbox_pred_loss_weight', default=0, type=float) # set to 0 to force regression on EPs only
  parser.add_argument('--triplet_bboxes_pred_loss_weight', default=10, type=float)
  parser.add_argument('--triplet_superboxes_pred_loss_weight', default=10, type=float)
  parser.add_argument('--triplet_mask_loss_weight', default=10, type=float)
  parser.add_argument('--predicate_pred_loss_weight', default=0, type=float) # DEPRECATED
  parser.add_argument('--perceptual_loss_weight', default=0, type=float)
  parser.add_argument('--grayscale_perceptual', action='store_true', help='Calculate perceptual loss with grayscale images')
  parser.add_argument('--log_perceptual', action='store_true', help='Take logarithm of perceptual loss')
  parser.add_argument('--extreme_pts_loss_weight', default=10, type=float)
  parser.add_argument('--extreme_to_mask', default=False, type=bool_flag)
  
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
  parser.add_argument('--checkpoint_every', default=10000, type=int)
  parser.add_argument('--check_val_metrics_every', default=10000, type=int)
  parser.add_argument('--output_dir', default=os.getcwd())
  parser.add_argument('--checkpoint_name', default='checkpoint')
  parser.add_argument('--checkpoint_start_from', default=None)
  parser.add_argument('--restore_from_checkpoint', default=True, type=bool_flag)

  # Logger output
  # parser.add_argument('--logdir_name', default='./logs')

  return validate_args(parser.parse_args())



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
  if args.checkpoint_start_from is not None:
    checkpoint = torch.load(args.checkpoint_start_from)
    kwargs = checkpoint['model_kwargs']
    model = Sg2ImModel(**kwargs)
    raw_state_dict = checkpoint['model_state']
    state_dict = {}
    for k, v in raw_state_dict.items():
      if k.startswith('module.'):
        k = k[7:]
      state_dict[k] = v
    model.load_state_dict(state_dict)
  else:
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
      'sg_context_dim': args.sg_context_dim,
      'sg_context_dim_d': args.sg_context_dim_d,
      'gcnn_pooling': args.gcnn_pooling,
      'triplet_box_net': args.triplet_box_net,
      'triplet_mask_size': args.triplet_mask_size,
      'triplet_embedding_size': args.triplet_embedding_size,
      'use_bbox_info': args.use_bbox_info,
      'triplet_superbox_net': args.triplet_superbox_net,
      'masks_to_triplet_mlp': args.masks_to_triplet_mlp,
      'masks_to_triplet_pixels': args.masks_to_triplet_pixels,
    }
    model = Sg2ImModel(**kwargs)
  return model, kwargs


def build_obj_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_obj_weight = args.d_obj_weight
  if d_weight == 0 or d_obj_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'vocab': vocab,
    'arch': args.d_obj_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
    'object_size': args.crop_size,
  }
  discriminator = AcCropDiscriminator(**d_kwargs)
  return discriminator, d_kwargs


def build_img_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_img_weight = args.d_img_weight
  if d_weight == 0 or d_img_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'arch': args.d_img_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding
  }
  
  ## MISSING SECTION - deleted because causing syntax errors 
  return discriminator, d_kwargs


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
    'seed': 0,  # randomize for train
    'heuristics_ordering' : args.heuristics_ordering
  }

  train_dset = None
  if args.coco_view_validation_error is False:
    train_dset = CocoSceneGraphDataset(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

  dset_kwargs['image_dir'] = args.coco_val_image_dir
  dset_kwargs['instances_json'] = args.coco_val_instances_json
  dset_kwargs['stuff_json'] = args.coco_val_stuff_json
  # extreme points
  dset_kwargs['extreme_pts_json'] = args.coco_instances_extreme_val_json
  dset_kwargs['max_samples'] = args.num_val_samples
  #  *deactivate* randomization for val (for consistent results)
  dset_kwargs['seed'] = args.random_seed
  val_dset = CocoSceneGraphDataset(**dset_kwargs)

  if args.coco_view_validation_error is True:
    print('Using val dataset at train dataset to view validation error')
    train_dset = val_dset
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    print('Valdiation dataset has %d images and %d objects' % (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))
 
  assert train_dset.vocab == val_dset.vocab
  vocab = json.loads(json.dumps(train_dset.vocab))

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
  }
  train_dset = VgSceneGraphDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

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
  train_loader = DataLoader(train_dset, **loader_kwargs)
  
  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)
  return vocab, train_loader, val_loader

def get_rel_score(args, t, loader, model):
  float_dtype = torch.FloatTensor
  long_dtype = torch.LongTensor
  num_samples = 0
  all_losses = defaultdict(list)
  total_iou = 0
  total_boxes = 0
  rel_score = 0
  
  with torch.no_grad():
    o_start = o_end = 0
    t_start = t_end = 0
    last_o_idx = last_t_idx = 0

    b = 0
    total_boxes = 0
    total_iou = 0
    for batch in loader:
      batch = [tensor.cuda() for tensor in batch]
      #batch = [tensor for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 9:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, triplet_masks, extreme_pts = batch
      #elif len(batch) == 7:
      #  imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1]

      objs = objs.detach()
      triples = triples.detach()
      # Run the model as it has been run during training
      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
      # imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
      imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triple_boxes_pred, triple_boxes_gt, triplet_masks_pred, boxes_pred_info, triplet_superboxes_pred, extreme_pts_pred = model_out
       
      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break

      # use extreme points to predict bboxes and extract bboxes GT
      #####################
      extreme_pts_pred = extreme_pts_pred.detach()
      extreme_pts = extreme_pts.detach()

      boxes[:,0] = extreme_pts[:,2]
      boxes[:,1] = extreme_pts[:,1]
      boxes[:,2] = extreme_pts[:,6]
      boxes[:,3] = extreme_pts[:,5]

      boxes_pred[:,0] = extreme_pts_pred[:,2]
      boxes_pred[:,1] = extreme_pts_pred[:,1]
      boxes_pred[:,2] = extreme_pts_pred[:,6]
      boxes_pred[:,3] = extreme_pts_pred[:,5]

      # (boxes_pred_info has same predicted boxes as boxes_pred)
      rel_score += relation_score(boxes_pred, boxes, masks_pred, masks, model.vocab)
      b += 1
      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)

    rel_score = rel_score/b
    avg_iou = total_iou / total_boxes
    return rel_score, avg_iou

def add_bbox_info(bboxes):
  # bbox = [x0 y0 x1 y1]
  h = bboxes[:,3] - bboxes[:,1]
  w = bboxes[:,2] - bboxes[:,0]
  log_hw = torch.log1p(h/w) #  log1p is better for small number
  bboxes_info = torch.cat([bboxes, log_hw[:,None]], dim=1)
  return bboxes_info

def check_model(args, t, loader, model, logger=None, log_tag='', write_images=False):
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor
  num_samples = 0
  all_losses = defaultdict(list)
  total_iou = 0
  total_boxes = 0
  with torch.no_grad():
    for batch in loader:
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 9:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, triplet_masks, extreme_pts = batch
      #elif len(batch) == 7:
        #imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1] 

      # Run the model as it has been run during training
      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
      # imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
      imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triplet_boxes_pred, triplet_boxes, triplet_masks_pred, boxes_pred_info, triplet_superboxes_pred, extreme_pts_pred = model_out

      # add additional information for GT boxes (hack to not change coco.py)
      boxes_info = None
      if args.use_bbox_info and boxes_pred_info is not None:
        boxes_info = add_bbox_info(boxes)
      triplet_superboxes = None
      # GT for triplet superbox
      triplet_superboxes = None
      if args.triplet_superbox_net and triplet_superboxes_pred is not None:
        # triplet_boxes = [ x1_0 y1_0 x1_1 y1_1 x2_0 y2_0 x2_1 y2_1]
        min_pts = triplet_boxes[:,:2]
        max_pts = triplet_boxes[:,6:8]
        triplet_superboxes = torch.cat([min_pts, max_pts], dim=1)
      
      # for layout model, we don't care about these
      #skip_pixel_loss = False
      #skip_perceptual_loss = False
      skip_pixel_loss = True 
      skip_perceptual_loss = True 

      # calculate all losses here
      total_loss, losses =  calculate_model_losses(
                                args, skip_pixel_loss, model, imgs, imgs_pred,
                                boxes, boxes_pred, masks, masks_pred,
                                boxes_info, boxes_pred_info,
                                predicates, predicate_scores,
                                triplet_boxes, triplet_boxes_pred, 
                                triplet_masks, triplet_masks_pred,
                                triplet_superboxes, triplet_superboxes_pred, 
                                extreme_pts, extreme_pts_pred,
                                skip_perceptual_loss)

      losses['total_loss'] = total_loss.item()

      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)

      for loss_name, loss_val in losses.items():
        all_losses[loss_name].append(loss_val)
      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break

    
    samples = {}
    samples['gt_img'] = imgs

    #pdb.set_trace()
    #model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
    #samples['gt_box_gt_mask'] = model_out[0]

    #model_out = model(objs, triples, obj_to_img, boxes_gt=boxes)
    #samples['gt_box_pred_mask'] = model_out[0]

    #model_out = model(objs, triples, obj_to_img)
    #samples['pred_box_pred_mask'] = model_out[0]
    
    #for k, v in samples.items():
    #  samples[k] = imagenet_deprocess_batch(v) 

    #  if logger is not None and write_images:
       #   #3. Log ground truth and predicted images
    #     with torch.no_grad():
    #       p_imgs = samples['gt_box_gt_mask'].detach() 
    #       gt_imgs = samples['gt_img'].detach() 
    #       p_gbox_pmsk_img = samples['gt_box_pred_mask'] 
    #       p_test_imgs = samples['pred_box_pred_mask'] 

    #     np_gt_imgs = [gt.cpu().numpy().transpose(1,2,0) for gt in gt_imgs]
    #     np_pred_imgs = [pred.cpu().numpy().transpose(1,2,0) for pred in p_imgs]
    #     np_gbox_pmsk_imgs = [pred.cpu().numpy().transpose(1,2,0) for pred in p_gbox_pmsk_img] 
    #     np_test_pred_imgs = [pred.cpu().numpy().transpose(1,2,0) for pred in p_test_imgs]  
    #     np_all_imgs = []
      
    #     for gt_img, gtb_gtm_img, gtb_pm_img, pred_img in zip(np_gt_imgs, np_pred_imgs, np_gbox_pmsk_imgs, np_test_pred_imgs):
    #       np_all_imgs.append((gt_img * 255.0).astype(np.uint8))
    #       np_all_imgs.append((gtb_gtm_img * 255.0).astype(np.uint8))
    #       np_all_imgs.append((gtb_pm_img * 255.0).astype(np.uint8)) 
    #       np_all_imgs.append((pred_img * 255.0).astype(np.uint8))  

    #     logger.image_summary(log_tag, np_all_imgs, t)
      ######################################################################### 

    mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
    avg_iou = total_iou / total_boxes

    masks_to_store = masks
    if masks_to_store is not None:
      masks_to_store = masks_to_store.data.cpu().clone()

    masks_pred_to_store = masks_pred
    if masks_pred_to_store is not None:
      masks_pred_to_store = masks_pred_to_store.data.cpu().clone()

  batch_data = {
    'objs': objs.detach().cpu().clone(),
    'boxes_gt': boxes.detach().cpu().clone(), 
    'masks_gt': masks_to_store,
    'triples': triples.detach().cpu().clone(),
    'obj_to_img': obj_to_img.detach().cpu().clone(),
    'triple_to_img': triple_to_img.detach().cpu().clone(),
    'boxes_pred': boxes_pred.detach().cpu().clone(),
    'masks_pred': masks_pred_to_store
  }
  out = [mean_losses, samples, batch_data, avg_iou]

  return tuple(out)


def calculate_model_losses(args, skip_pixel_loss, model, img, img_pred,
                           bbox, bbox_pred, masks, masks_pred,
                           boxes_info, boxes_pred_info,
                           predicates, predicate_scores, 
                           triplet_bboxes, triplet_bboxes_pred, 
                           triplet_masks, triplet_masks_pred,
                           triplet_superboxes, triplet_superboxes_pred,
                           extreme_pts, extreme_pts_pred,
                           skip_perceptual_loss,
                           perceptual_extractor=None):  #FeatureExtractor(requires_grad=False).cuda()):
  total_loss = torch.zeros(1).to(img)
  losses = {}

  # for layout model, L1 pixel loss not used
  #l1_pixel_weight = args.l1_pixel_loss_weight
  #if skip_pixel_loss:
  #  l1_pixel_weight = 0

  # l1_pixel_loss = F.l1_loss(img_pred, img)
  # total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
  #                       l1_pixel_weight)

  # extreme point regression 
  if args.extreme_pts_loss_weight > 0:
    loss_extreme_pts = F.mse_loss(extreme_pts_pred, extreme_pts)
    total_loss = add_loss(total_loss, loss_extreme_pts, losses, 'extreme_pts_loss',
                            args.extreme_pts_loss_weight)
 
  #  bounding box regression loss with bbox 'meta' info added 
  if args.use_bbox_info and boxes_info is not None and boxes_pred_info is not None:
    loss_bbox = F.mse_loss(boxes_pred_info, boxes_info)
  else:
    loss_bbox = F.mse_loss(bbox_pred, bbox)
  if args.bbox_pred_loss_weight > 0:
    total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                        args.bbox_pred_loss_weight)

  # superbox regression for triplets
  if args.triplet_superboxes_pred_loss_weight > 0 and triplet_superboxes is not None and triplet_superboxes_pred is not None:
    loss_triplet_superboxes = F.mse_loss(triplet_superboxes_pred, triplet_superboxes)
    total_loss = add_loss(total_loss, loss_triplet_superboxes, losses, 'triplet_superboxes_pred', 
                        args.triplet_superboxes_pred_loss_weight)

  #  bounding boxes regression for triplets
  if args.triplet_bboxes_pred_loss_weight > 0 and triplet_bboxes is not None and triplet_bboxes_pred is not None:
    loss_triplet_bboxes = F.mse_loss(triplet_bboxes_pred, triplet_bboxes)
    total_loss = add_loss(total_loss, loss_triplet_bboxes, losses, 'triplet_bboxes_pred', 
                        args.triplet_bboxes_pred_loss_weight)

  if args.triplet_mask_loss_weight > 0 and triplet_masks is not None and triplet_masks_pred is not None:
    # for multiclass (subj/obj/background) labels
    triplet_mask_loss = F.cross_entropy(triplet_masks_pred, triplet_masks.long())
    #triplet_mask_loss = F.binary_cross_entropy(triplet_masks_pred, triplet_masks.float())
    total_loss = add_loss(total_loss, triplet_mask_loss, losses, 'triplet_mask_loss',
                          args.triplet_mask_loss_weight)

  if args.predicate_pred_loss_weight > 0:
    loss_predicate = F.cross_entropy(predicate_scores, predicates)
    total_loss = add_loss(total_loss, loss_predicate, losses, 'predicate_pred',
                          args.predicate_pred_loss_weight)

  if args.mask_loss_weight > 0 and masks is not None and masks_pred is not None:
    mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
    total_loss = add_loss(total_loss, mask_loss, losses, 'mask_loss',
                          args.mask_loss_weight)

  #########################
  #perceptual_weight = args.perceptual_loss_weight
  if skip_perceptual_loss:
    perceptual_weight = 0
   
  #if perceptual_weight > 0:  
  #  with torch.no_grad():
  #    img_p = perc_process_batch(img.detach(), rescale=True, grayscale=args.grayscale_perceptual)
  #    img_pred_p = perc_process_batch(img_pred.detach(), rescale=True, grayscale=args.grayscale_perceptual)
  #
  #    real = perceptual_extractor(img_p)
  #    fake = perceptual_extractor(img_pred_p)
  #
  #    vgg_perceptual_loss = F.mse_loss(fake.relu1_2, real.relu1_2)
  #    vgg_perceptual_loss += F.mse_loss(fake.relu2_2, real.relu2_2)
  #    vgg_perceptual_loss += F.mse_loss(fake.relu3_3, real.relu3_3)
  #    vgg_perceptual_loss += F.mse_loss(fake.relu4_3, real.relu4_3)

   # total_loss = add_loss(total_loss, vgg_perceptual_loss, losses, 'perceptual_loss',
   #                       perceptual_weight, logarithm=args.log_perceptual)
  #########################  
  return total_loss, losses


def main(args):
  print(args)
  check_args(args)
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor

  vocab, train_loader, val_loader = build_loaders(args)
  model, model_kwargs = build_model(args, vocab)
  model.type(float_dtype)
  print(model)

  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
    print('Created %s' %args.output_dir)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  vgg_featureExractor = FeatureExtractor(requires_grad=False).cuda()
  ## add code for training visualization
  logger = Logger(args.output_dir)

  obj_discriminator, d_obj_kwargs = build_obj_discriminator(args, vocab)
  img_discriminator, d_img_kwargs = build_img_discriminator(args, vocab)
  gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)
  
  if args.matching_aware_loss and args.sg_context_dim > 0:
    gan_g_matching_aware_loss, gan_d_matching_aware_loss = get_gan_losses('matching_aware_gan')

  ### quick hack
  obj_discriminator = None
  img_discriminator = None


  ############
  if obj_discriminator is not None:
    obj_discriminator.type(float_dtype)
    obj_discriminator.train()
    print(obj_discriminator)
    optimizer_d_obj = torch.optim.Adam(obj_discriminator.parameters(),
                                       lr=args.learning_rate)

  if img_discriminator is not None:
    img_discriminator.type(float_dtype)
    img_discriminator.train()
    print(img_discriminator)
    optimizer_d_img = torch.optim.Adam(img_discriminator.parameters(),
                                       lr=args.learning_rate)

  restore_path = None
  if args.checkpoint_start_from is not None:
    restore_path = args.checkpoint_start_from
  elif args.restore_from_checkpoint:
    restore_path = '%s_with_model.pt' % args.checkpoint_name
    restore_path = os.path.join(args.output_dir, restore_path)
  if restore_path is not None and os.path.isfile(restore_path):
    print('Restoring from checkpoint:')
    print(restore_path)
    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])

    if obj_discriminator is not None:
      obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
      optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])

    if img_discriminator is not None:

      img_discriminator.load_state_dict(checkpoint['d_img_state'])
      optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])

    t = checkpoint['counters']['t']
    if 0 <= args.eval_mode_after <= t:
      model.eval()
    else:
      model.train()
    epoch = checkpoint['counters']['epoch']
  else:
    t, epoch = 0, 0
    checkpoint = {
      'args': args.__dict__,
      'vocab': vocab,
      'model_kwargs': model_kwargs,
      'd_obj_kwargs': d_obj_kwargs,
      'd_img_kwargs': d_img_kwargs,
      'losses_ts': [],
      'losses': defaultdict(list),
      'd_losses': defaultdict(list),
      'checkpoint_ts': [],
      'train_batch_data': [], 
      'train_samples': [],
      'train_iou': [],
      'val_batch_data': [], 
      'val_samples': [],
      'val_losses': defaultdict(list),
      'val_iou': [], 
      'norm_d': [], 
      'norm_g': [],
      'counters': {
        't': None,
        'epoch': None,
      },
      'model_state': None, 'model_best_state': None, 'optim_state': None,
      'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
      'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
      'best_t': [],
    }


  while True:
    if t >= args.num_iterations:
      break
    epoch += 1
    print('Starting epoch %d' % epoch)
    
    for batch in train_loader:
      if t == args.eval_mode_after:
        print('switching to eval mode')
        model.eval()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
      t += 1
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 9:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, triplet_masks, extreme_pts = batch
      #elif len(batch) == 7:
      #  imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      else:
        assert False
      predicates = triples[:, 1]


      with timeit('forward', args.timing):
        model_boxes = boxes
        model_masks = masks
        model_out = model(objs, triples, obj_to_img,
                          boxes_gt=model_boxes, masks_gt=model_masks
                          )
        # imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
        imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triplet_boxes_pred, triplet_boxes, triplet_masks_pred, boxes_pred_info, triplet_superboxes_pred, extreme_pts_pred = model_out
        # boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores = model_out
     
      # add additional information for GT boxes (hack to not change coco.py)
      boxes_info = None
      if args.use_bbox_info and boxes_pred_info is not None:
        boxes_info = add_bbox_info(boxes) 
      # GT for triplet superbox
      triplet_superboxes = None
      if args.triplet_superbox_net and triplet_superboxes_pred is not None:
        # triplet_boxes = [ x1_0 y1_0 x1_1 y1_1 x2_0 y2_0 x2_1 y2_1]
        min_pts = triplet_boxes[:,:2]
        max_pts = triplet_boxes[:,6:8]
        triplet_superboxes = torch.cat([min_pts, max_pts], dim=1)

      with timeit('loss', args.timing):
        # Skip the pixel loss if using GT boxes
        #skip_pixel_loss = (model_boxes is None)
        skip_pixel_loss = True 
        # Skip the perceptual loss if using GT boxes
        #skip_perceptual_loss = (model_boxes is None)
        skip_perceptual_loss = True 

        if args.perceptual_loss_weight:
          total_loss, losses =  calculate_model_losses(
                                  args, skip_pixel_loss, model, imgs, mgs_pred,
                                  boxes, boxes_pred, masks, masks_pred,
                                  boxes_info, boxes_pred_info,
                                  predicates, predicate_scores, 
                                  triplet_boxes, triplet_boxes_pred, 
                                  triplet_masks, triplet_masks_pred, 
                                  triplet_superboxes, triplet_superboxes_pred,
                                  extreme_pts, extreme_pts_pred,
                                  skip_perceptual_loss,
                                  perceptual_extractor=vgg_featureExractor)
        else:
          total_loss, losses =  calculate_model_losses(
                                    args, skip_pixel_loss, model, imgs, imgs_pred,
                                    boxes, boxes_pred, masks, masks_pred,
                                    boxes_info, boxes_pred_info,
                                    predicates, predicate_scores, 
                                    triplet_boxes, triplet_boxes_pred, 
                                    triplet_masks, triplet_masks_pred,
                                    triplet_superboxes, triplet_superboxes_pred,
                                    extreme_pts, extreme_pts_pred,
                                    skip_perceptual_loss)  
          #total_loss, losses =  calculate_model_losses(
          #                          args, skip_pixel_loss, model, imgs, imgs_pred,
          #                          boxes, boxes_pred, masks, masks_pred,
          #                          predicates, predicate_scores, 

      if obj_discriminator is not None:
        scores_fake, ac_loss = obj_discriminator(imgs_pred, objs, boxes, obj_to_img)
        total_loss = add_loss(total_loss, ac_loss, losses, 'ac_loss',
                              args.ac_loss_weight)
        weight = args.discriminator_loss_weight * args.d_obj_weight
        total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                              'g_gan_obj_loss', weight)

      if img_discriminator is not None:
        weight = args.discriminator_loss_weight * args.d_img_weight
      
        # scene_graph context by pooled GCNN features
        if args.sg_context_dim > 0:
          ## concatenate => imgs, (layout_embedding), sg_context_pred
          if args.layout_for_discrim == 1:
            discrim_pred = torch.cat([imgs_pred, layout, sg_context_pred_d], dim=1) 
          else:
            discrim_pred = torch.cat([imgs_pred, sg_context_pred_d], dim=1)  
          
          if args.matching_aware_loss:
            # shuffle sg_context_p to use addional fake examples with real-images
            matching_aware_size = sg_context_pred_d.size()[0]
            s_sg_context_pred_d = sg_context_pred_d[torch.randperm(matching_aware_size)]  
            if args.layout_for_discrim == 1:
              match_aware_discrim_pred = torch.cat([imgs, layout, s_sg_context_pred_d], dim=1) 
            else: 
              match_aware_discrim_pred = torch.cat([imgs, s_sg_context_pred_d], dim=1 )           
            discrim_pred = torch.cat([discrim_pred, match_aware_discrim_pred], dim=0)         
          
          scores_fake = img_discriminator(discrim_pred)         
          if args.matching_aware_loss:
            total_loss = add_loss(total_loss, gan_g_matching_aware_loss(scores_fake), losses,
                            'g_gan_img_loss', weight)
          else:
            total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                              'g_gan_img_loss', weight)
        else:
          scores_fake = img_discriminator(imgs_pred)
          #weight = args.discriminator_loss_weight * args.d_img_weight
          total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                                'g_gan_img_loss', weight)

      losses['total_loss'] = total_loss.item()
      if not math.isfinite(losses['total_loss']):
        print('WARNING: Got loss = NaN, not backpropping')
        continue

      optimizer.zero_grad()
      with timeit('backward', args.timing):
        total_loss.backward()
      optimizer.step()
      total_loss_d = None
      ac_loss_real = None
      ac_loss_fake = None
      d_losses = {}
      
      if obj_discriminator is not None:
        d_obj_losses = LossManager()
        imgs_fake = imgs_pred.detach()
        scores_fake, ac_loss_fake = obj_discriminator(imgs_fake, objs, boxes, obj_to_img)
        scores_real, ac_loss_real = obj_discriminator(imgs, objs, boxes, obj_to_img)

        d_obj_gan_loss = gan_d_loss(scores_real, scores_fake)
        d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
        d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
        d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

        optimizer_d_obj.zero_grad()
        d_obj_losses.total_loss.backward()
        optimizer_d_obj.step()


      if img_discriminator is not None:
        d_img_losses = LossManager()
               
        imgs_fake = imgs_pred.detach()

        if args.sg_context_dim_d > 0:
          sg_context_p = sg_context_pred_d.detach()  
        
        layout_p = layout.detach()
        # layout_gt_p = layout_gt.detach()

        ## concatenate=> imgs_fake, (layout_embedding), sg_context_pred
        if args.sg_context_dim > 0:
          if args.layout_for_discrim:
            discrim_fake = torch.cat([imgs_fake, layout_p, sg_context_p], dim=1 )  
            discrim_real = torch.cat([imgs, layout_p, sg_context_p], dim=1 ) 
            # discrim_real = torch.cat([imgs, layout_gt_p, sg_context_p], dim=1 ) 
          else:   
            discrim_fake = torch.cat([imgs_fake, sg_context_p], dim=1 ) 
            discrim_real = torch.cat([imgs, sg_context_p], dim=1 )   
              
          if args.matching_aware_loss:
            # shuffle sg_context_p to use addional fake examples with real-images
            matching_aware_size = sg_context_p.size()[0]
            s_sg_context_p = sg_context_p[torch.randperm(matching_aware_size)]
            # s_sg_context_p = sg_context_p[torch.randperm(args.batch_size)]
            if args.layout_for_discrim:
              match_aware_discrim_fake = torch.cat([imgs, layout_p, s_sg_context_p], dim=1 ) 
            else:
              match_aware_discrim_fake = torch.cat([imgs, s_sg_context_p], dim=1 ) 
            discrim_fake = torch.cat([discrim_fake, match_aware_discrim_fake], dim=0)   

          scores_fake = img_discriminator(discrim_fake) 
          scores_real = img_discriminator(discrim_real)

          if args.matching_aware_loss:
            d_img_gan_loss = gan_d_matching_aware_loss(scores_real, scores_fake)
          else:
            d_img_gan_loss = gan_d_loss(scores_real, scores_fake)
        else:
          # imgs_fake = imgs_pred.detach()
          scores_fake = img_discriminator(imgs_fake)
          scores_real = img_discriminator(imgs)

        if args.matching_aware_loss:
          d_img_gan_loss = gan_d_matching_aware_loss(scores_real, scores_fake)
        else:
          d_img_gan_loss = gan_d_loss(scores_real, scores_fake)
          
        d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')
        
        optimizer_d_img.zero_grad()
        d_img_losses.total_loss.backward()
        optimizer_d_img.step()

      # report intermediary values to stdout
      if t % args.print_every == 0:
        print('t = %d / %d' % (t, args.num_iterations))
        for name, val in losses.items():
          print(' G [%s]: %.4f' % (name, val))
          checkpoint['losses'][name].append(val)
        checkpoint['losses_ts'].append(t)

        if obj_discriminator is not None:
          for name, val in d_obj_losses.items():
            print(' D_obj [%s]: %.4f' % (name, val))
            checkpoint['d_losses'][name].append(val)

        if img_discriminator is not None:
          for name, val in d_img_losses.items():
            print(' D_img [%s]: %.4f' % (name, val))
            checkpoint['d_losses'][name].append(val)

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        for name, val in losses.items():
            logger.scalar_summary(name, val, t)
        if obj_discriminator is not None:    
          for name, val in d_obj_losses.items():
              logger.scalar_summary(name, val, t)           
        if img_discriminator is not None:
          for name, val in d_img_losses.items():
              logger.scalar_summary(name, val, t)   
        logger.scalar_summary('score', t, t)

        if t % args.check_val_metrics_every == 0 and t > args.print_every:
          print('Checking val metrics')
          rel_score, avg_iou = get_rel_score(args, t, val_loader, model)
          logger.scalar_summary('relation_score', rel_score, t) 
          logger.scalar_summary('avg_IoU', avg_iou, t) 
          print(t, ': val relation score = ', rel_score)
          print(t, ': IoU = ', avg_iou)
          # checks entire val dataset
          val_results = check_model(args, t, val_loader, model, logger=logger, log_tag='Val', write_images=False)
          v_losses, v_samples, v_batch_data, v_avg_iou = val_results
          logger.scalar_summary('val_total_loss', v_losses['total_loss'], t)
      
      if t % args.checkpoint_every == 0:
        print('checking on train')
        train_results = check_model(args, t, train_loader, model, logger=logger, log_tag='Train', write_images=False)
        t_losses, t_samples, t_batch_data, t_avg_iou = train_results

        checkpoint['train_batch_data'].append(t_batch_data)
        checkpoint['train_samples'].append(t_samples)
        checkpoint['checkpoint_ts'].append(t)
        checkpoint['train_iou'].append(t_avg_iou)

        print('checking on val')
        val_results = check_model(args, t, val_loader, model, logger=logger, log_tag='Validation', write_images=True)

        val_losses, val_samples, val_batch_data, val_avg_iou = val_results
        checkpoint['val_samples'].append(val_samples)
        checkpoint['val_batch_data'].append(val_batch_data)
        checkpoint['val_iou'].append(val_avg_iou)
        
        print('train iou: ', t_avg_iou)
        print('val iou: ', val_avg_iou)

        for k, v in val_losses.items():
          checkpoint['val_losses'][k].append(v)
        checkpoint['model_state'] = model.state_dict()

        if obj_discriminator is not None:
          checkpoint['d_obj_state'] = obj_discriminator.state_dict()
          checkpoint['d_obj_optim_state'] = optimizer_d_obj.state_dict()

        if img_discriminator is not None:
          checkpoint['d_img_state'] = img_discriminator.state_dict()
          checkpoint['d_img_optim_state'] = optimizer_d_img.state_dict()

        checkpoint['optim_state'] = optimizer.state_dict()
        checkpoint['counters']['t'] = t
        checkpoint['counters']['epoch'] = epoch
        checkpoint_path = os.path.join(args.output_dir,
                              '%s_with_model_%d.pt' %(args.checkpoint_name, t) 
                              #'%s_with_model.pt' %args.checkpoint_name
                              )
        print('Saving checkpoint to ', checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

        # Save another checkpoint without any model or optim state
        checkpoint_path = os.path.join(args.output_dir,
                              '%s_no_model.pt' % args.checkpoint_name)
        key_blacklist = ['model_state', 'optim_state', 'model_best_state',
                         'd_obj_state', 'd_obj_optim_state', 'd_obj_best_state',
                         'd_img_state', 'd_img_optim_state', 'd_img_best_state']
        small_checkpoint = {}
        for k, v in checkpoint.items():
          if k not in key_blacklist:
            small_checkpoint[k] = v
        torch.save(small_checkpoint, checkpoint_path)


if __name__ == '__main__':
  # args = parser.parse_args()
  # main(args)
  main(parse_args())

