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


from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator, CondGANPatchDiscriminator, CondGANDiscriminator
from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard, relation_score

from sg2im.model import Sg2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager

####
from sg2im.perceptual_loss import FeatureExtractor
from sg2im.data.utils import imagenet_deprocess_batch
from sg2im.data.utils import perc_process_batch
#from sg2im.logger import Logger

from imageio import imwrite
import sg2im.vis as vis

import pdb


#torch.backends.cudnn.benchmark = True

#COCO_DIR = os.path.expanduser('/dataset/coco_stuff')
COCO_DIR = os.path.expanduser('./dataset/coco_stuff')
VG_DIR = os.path.expanduser('./dataset/vg')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=100000, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'test.h5')) #val.h5
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
parser.add_argument('--random_seed', default=42, type=int)  #For comparative results/debug

# Generator options
parser.add_argument('--mask_size', default=0, type=int) # Set this to 0 to use no masks
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
parser.add_argument('--output_dir', default=os.getcwd())

parser.add_argument('--checkpoint', default='sg2im-models/vg64.pt')
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])




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
  }
  model = Sg2ImModel(**kwargs)
  return model, kwargs


def build_coco_dsets(args):
  dset_kwargs = {
    'image_dir': args.coco_train_image_dir,
    'instances_json': args.coco_train_instances_json,
    'stuff_json': args.coco_train_stuff_json,
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
    'seed': args.random_seed,
  }
  train_dset = CocoSceneGraphDataset(**dset_kwargs)
  num_objs = train_dset.total_objects()
  num_imgs = len(train_dset)
  print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
  print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

  dset_kwargs['image_dir'] = args.coco_val_image_dir
  dset_kwargs['instances_json'] = args.coco_val_instances_json
  dset_kwargs['stuff_json'] = args.coco_val_stuff_json
  dset_kwargs['max_samples'] = args.num_val_samples
  val_dset = CocoSceneGraphDataset(**dset_kwargs)

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
    'seed': args.random_seed,
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

  ###################
  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
    print('Created %s' %args.output_dir)

  img_dir = args.output_dir+'/img_dir'

  if not os.path.isdir(img_dir):
    os.mkdir(img_dir)
    print('Created %s' %img_dir) 
  ################## 
  rel_score = 0
  with torch.no_grad():
    o_start = o_end = 0
    t_start = t_end = 0
    last_o_idx = last_t_idx = 0

    b = 0
    total_boxes = 0
    total_iou = 0
    for batch in loader:
      #batch = [tensor.cuda() for tensor in batch]
      batch = [tensor for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 7:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1] 

      objs = objs.detach()
      triples = triples.detach()
      # Run the model as it has been run during training
      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
      # imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
      imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores = model_out

      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break     

      rel_score += relation_score(boxes_pred, boxes, masks_pred, masks, model.vocab) 
      b += 1

      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)

    print (b)
    print (total_boxes)


    rel_score = rel_score/b
    avg_iou = total_iou / total_boxes

    # print('rel score:' rel_score)
    # print('avg iou:' avg_iou)
 

    return rel_score, avg_iou



def check_model_predicate_debug(args, t, loader, model, logger=None, log_tag='', write_images=False):
  # float_dtype = torch.cuda.FloatTensor
  # long_dtype = torch.cuda.LongTensor
  float_dtype = torch.FloatTensor
  long_dtype = torch.LongTensor
  num_samples = 0
  all_losses = defaultdict(list)
  total_iou = 0
  total_boxes = 0

  sane_sg_im_list = [0,4,9,18,22,30,32,42,54,58,63,66,69,74,76,80,82,88,96,100,101,107,111,116,136,147,168,170,175,
                    195,196,204,211,214,218,234,241,244,246,261,262,268,288,291,313,339,343,369,374,376,402,407,410,
                    422,425,431,440,455,461,463,465,467,468,471,474,489,493]

  ###################
  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
    print('Created %s' %args.output_dir)

  img_dir = args.output_dir+'/img_dir'

  if not os.path.isdir(img_dir):
    os.mkdir(img_dir)
    print('Created %s' %img_dir) 
  ##################  

  t = 0
  t1 = 0
  bt = 0
  with torch.no_grad():
    o_start = o_end = 0
    t_start = t_end = 0
    last_o_idx = last_t_idx = 0
    for batch in loader:
      batch = [tensor for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 7:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1] 

      ############################
      sg_objs = objs[obj_to_img==0]
      sg_rels = triples[triple_to_img==0]
      pdb.set_trace()
      print(sum(predicates.numpy()))
      ############################  
      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break

      if draw_scene_graph == True:
        ## Draw scene graph
        tot_obj = 0
        for b_t in range(imgs.size(0)):
          sg_objs = objs[obj_to_img==b_t]
          sg_rels = triples[triple_to_img==b_t]
          sg_img = vis.draw_scene_graph_temp(sg_objs, sg_rels, tot_obj, vocab=model.vocab) 
          # sg_img = vis.draw_scene_graph(sg_objs, sg_rels, vocab=model.vocab)  
          sg_img_path = os.path.join(img_dir, '%06d_sg.png' % t1)
          imwrite(sg_img_path, sg_img) 

          tot_obj = tot_obj + len(sg_objs) 
          t1 = t1+1 






def check_model(args, t_2, loader, model, device, logger=None, log_tag='', write_images=False):
  # float_dtype = torch.cuda.FloatTensor
  # long_dtype = torch.cuda.LongTensor

  if device == torch.device('cpu'):
    float_dtype = torch.FloatTensor
    long_dtype = torch.LongTensor
  else:
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor

  num_samples = 0
  all_losses = defaultdict(list)
  total_iou = 0
  total_boxes = 0

  draw_scene_graph = True

  sane_sg_im_list = [0,4,9,18,22,30,32,42,54,58,63,66,69,74,76,80,82,88,96,100,101,107,111,
                    116,136,147,168,170,175,195,196,204,211,214,218,234,241,244,246,261,262,
                    268,288,291,313,339,343,369,374,376,402,407,410,422,425,431,440,455,461,
                    463,465,467,468,471,474,489,493]

  ###################
  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
    print('Created %s' %args.output_dir)

  img_dir = args.output_dir+'/img_dir'

  if not os.path.isdir(img_dir):
    os.mkdir(img_dir)
    print('Created %s' %img_dir) 
  ##################  

  t = 0
  t1 = 0
  bt = 0

  with torch.no_grad():
    o_start = o_end = 0
    t_start = t_end = 0
    last_o_idx = last_t_idx = 0
    for batch in loader:
      #batch = [tensor.cuda() for tensor in batch]
      batch = [tensor for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 7:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1] 


      objs = objs.detach()
      triples = triples.detach()
      # Run the model as it has been run during training
      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
      # imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
      imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores = model_out

      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break

      samples = {}
      samples['gt_img'] = imgs


      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)  
     
      if write_images:
        model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
        samples['gt_box_gt_mask'] = model_out[0]

        model_out = model(objs, triples, obj_to_img, boxes_gt=boxes)
        samples['gt_box_pred_mask'] = model_out[0]

      ##############################################
      # import pdb
      # pdb.set_trace()
      # num_boxes=len(boxes)
      # model_out = model(objs, triples, obj_to_img, boxes_gt=scaled_boxes)
      # samples['gt_scaled_box_pred_mask'] = model_out[0]
      ##############################################

        model_out = model(objs, triples, obj_to_img)
        samples['pred_box_pred_mask'] = model_out[0]

        layout_preds = {}
        layout_preds['pred_boxes'] = model_out[5]
        layout_preds['pred_masks'] = model_out[6]
        
        for k, v in samples.items():
          samples[k] = imagenet_deprocess_batch(v) 


        #3. Log ground truth and predicted images
        with torch.no_grad():
          gt_imgs = samples['gt_img'].detach() 
          p_gbox_pmsk_img = samples['gt_box_pred_mask'].detach() 
          p_test_imgs = samples['pred_box_pred_mask'].detach() 
          
          p_test_boxes = layout_preds['pred_boxes']
          p_test_masks = layout_preds['pred_masks']


        np_gt_imgs = [gt.cpu().numpy().transpose(1,2,0) for gt in gt_imgs]
        np_gbox_pmsk_imgs = [pred.cpu().numpy().transpose(1,2,0) for pred in p_gbox_pmsk_img] 
        np_test_pred_imgs = [pred.cpu().numpy().transpose(1,2,0) for pred in p_test_imgs]  

        pred_layout_boxes = p_test_boxes 
        pred_layout_masks = p_test_masks 
        np_all_imgs = []

        # Overlay box on images
        pred_layout_boxes_t = pred_layout_boxes.detach()
        # overlaid_images = vis.overlay_boxes(np_test_pred_imgs, model.vocab, objs_vec, layout_boxes_t, obj_to_img, W=64, H=64)   
        overlaid_images = vis.overlay_boxes(np_test_pred_imgs, model.vocab, objs_vec, pred_layout_boxes_t, obj_to_img, W=64, H=64) 
        
        # # # draw the layout
        # layouts_gt = vis.debug_layout_mask(model.vocab, objs_vec, layout_boxes, layout_masks, obj_to_img, W=128, H=128)
        # layouts_pred = vis.debug_layout_mask(model.vocab, objs_vec, pred_layout_boxes, pred_layout_masks, obj_to_img, W=128, H=128)

        for gt_img, gtb_pm_img, pred_img, overlaid in zip(np_gt_imgs, np_gbox_pmsk_imgs, np_test_pred_imgs, overlaid_images):
        # for gt_img, gtb_gtm_img, gtb_pm_img, pred_img, gt_layout_img, pred_layout_img, overlaid in zip(np_gt_imgs, np_pred_imgs, np_gbox_pmsk_imgs, np_test_pred_imgs, layouts_gt, layouts_pred, overlaid_images):
          img_path = os.path.join(img_dir, '%06d_gt_img.png' % t)
          imwrite(img_path, gt_img)

          img_path = os.path.join(img_dir, '%06d_gtb_pm_img.png' % t)
          imwrite(img_path, gtb_pm_img)

          img_path = os.path.join(img_dir, '%06d_pred_img.png' % t)
          imwrite(img_path, pred_img)

          overlaid_path = os.path.join(img_dir, '%06d_overlaid.png' % t)  
          imwrite(overlaid_path, overlaid)

          t=t+1
          print(t) 


        
        ## Draw scene graph
        tot_obj = 0
        for b_t in range(imgs.size(0)):
          sg_objs = objs[obj_to_img==b_t]
          sg_rels = triples[triple_to_img==b_t]

          if draw_scene_graph == True:
            sg_img = vis.draw_scene_graph_temp(sg_objs, sg_rels, tot_obj, vocab=model.vocab) 
            sg_img_path = os.path.join(img_dir, '%06d_sg.png' % t1)
            imwrite(sg_img_path, sg_img)   

          if t1 in sane_sg_im_list:
            vis.draw_scene_graph_json(t1, sg_objs, sg_rels, tot_obj, vocab=model.vocab) 

          # sg_img = vis.draw_scene_graph(sg_objs, sg_rels, vocab=model.vocab)  
          

          tot_obj = tot_obj + len(sg_objs) 
          t1 = t1+1 


    # mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
    # avg_iou = total_iou / total_boxes

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
  #out = [mean_losses, samples, batch_data, avg_iou]
  out = [samples]

  ####################
  avg_iou = total_iou / total_boxes
  print(avg_iou.numpy())
  ###################

  return tuple(out)


def calculate_model_losses(args, skip_pixel_loss, model, img, img_pred,
                           bbox, bbox_pred, masks, masks_pred,
                           predicates, predicate_scores, 
                           skip_perceptual_loss,
                           perceptual_extractor=None):  #FeatureExtractor(requires_grad=False).cuda()):
  total_loss = torch.zeros(1).to(img)
  losses = {}

  l1_pixel_weight = args.l1_pixel_loss_weight
  if skip_pixel_loss:
    l1_pixel_weight = 0

  l1_pixel_loss = F.l1_loss(img_pred, img)
  total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                        l1_pixel_weight)
  loss_bbox = F.mse_loss(bbox_pred, bbox)
  total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                        args.bbox_pred_loss_weight)

  if args.predicate_pred_loss_weight > 0:
    loss_predicate = F.cross_entropy(predicate_scores, predicates)
    total_loss = add_loss(total_loss, loss_predicate, losses, 'predicate_pred',
                          args.predicate_pred_loss_weight)

  if args.mask_loss_weight > 0 and masks is not None and masks_pred is not None:
    mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
    total_loss = add_loss(total_loss, mask_loss, losses, 'mask_loss',
                          args.mask_loss_weight)

  #########################
  perceptual_weight = args.perceptual_loss_weight
  if skip_perceptual_loss:
    perceptual_weight = 0
  
  if perceptual_weight > 0:  
    with torch.no_grad():
      img_p = perc_process_batch(img.detach(), rescale=True, grayscale=args.grayscale_perceptual)
      img_pred_p = perc_process_batch(img_pred.detach(), rescale=True, grayscale=args.grayscale_perceptual)

      real = perceptual_extractor(img_p)
      fake = perceptual_extractor(img_pred_p)

      vgg_perceptual_loss = F.mse_loss(fake.relu1_2, real.relu1_2)
      vgg_perceptual_loss += F.mse_loss(fake.relu2_2, real.relu2_2)
      vgg_perceptual_loss += F.mse_loss(fake.relu3_3, real.relu3_3)
      vgg_perceptual_loss += F.mse_loss(fake.relu4_3, real.relu4_3)

    total_loss = add_loss(total_loss, vgg_perceptual_loss, losses, 'perceptual_loss',
                          perceptual_weight, logarithm=args.log_perceptual)
  #########################  
  return total_loss, losses


def main(args):

  if args.device == 'cpu':
    device = torch.device('cpu')
  elif args.device == 'gpu':
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
      print('WARNING: CUDA not available; falling back to CPU')
      device = torch.device('cpu')

  # Load the model, with a bit of care in case there are no GPUs
  map_location = 'cpu' if device == torch.device('cpu') else None
  checkpoint = torch.load(args.checkpoint, map_location=map_location)
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'], strict=False)
  model.eval()
  model.to(device)

  vocab, train_loader, val_loader = build_loaders(args)

  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
    print('Created %s' %args.output_dir)

  ## add code for validation visualization
  #logger = Logger(args.output_dir)
  logger = None
  t = 1 
  with timeit('forward', args.timing):   
    print('checking on val')


    import json 
    with open('vg_captions_500.json', 'w') as f: 
      f.write('[')

    # check_model_predicate_debug(args, t, val_loader, model, logger=logger, log_tag='Validation', write_images=True)
    val_results = check_model(args, t, val_loader, model, device, logger=logger, log_tag='Validation', write_images=True)

    # rel_score, avg_iou = get_rel_score(args, t, val_loader, model)
    # print ('relation score: ', rel_score)
    # print ('average iou: ', avg_iou)
    # val_losses, val_avg_iou = val_results
    # print('val iou: ', val_avg_iou)

    with open('vg_captions_500.json', 'a') as f: 
      f.write(']')

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

