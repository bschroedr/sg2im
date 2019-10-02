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
import pdb
import pprint
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

#from sg2im.data.coco_ss import CocoSceneGraphDataset, coco_collate_fn
#from sg2im.data.coco_contour import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator, CondGANPatchDiscriminator, CondGANDiscriminator
#from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard

#from sg2im.model_layout_contour import Sg2ImModel
from sg2im.model_layout import Sg2ImModel
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

COCO_DIR = os.path.expanduser('/dataset/coco_stuff')
#COCO_DIR = os.path.expanduser('/Users/brigitsc/sandbox/sg2im/datasets/coco')

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

# triplet options
parser.add_argument('--triplet_box_net', default=False, type=int)
parser.add_argument('--triplet_mask_size', default=0, type=int)
# embedding for contextual search
parser.add_argument('--triplet_embedding_size', default=0, type=int)


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
parser.add_argument('--output_dir', default=os.getcwd())

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
  pdb.set_trace()
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
    'seed': args.random_seed   
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
  dset_kwargs['max_samples'] = args.num_val_samples
  #  *deactivate* randomization for val (for consistent results)
  dset_kwargs['seed'] = args.random_seed
  val_dset = CocoSceneGraphDataset(**dset_kwargs)

  #assert train_dset.vocab == val_dset.vocab
  #vocab = json.loads(json.dumps(train_dset.vocab))
  vocab = json.loads(json.dumps(val_dset.vocab))

  return vocab, train_dset, val_dset


def build_loaders(args):
  if args.dataset == 'coco':
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

  t = 0
  # relationship (triplet) database
  triplet_db = dict()

  # iterate over all batches of images
  with torch.no_grad():
    for batch in loader:

      # TODO: HERE
      if torch.cuda.is_available():
        batch = [tensor.cuda() for tensor in batch]
      else:
        batch = [tensor for tensor in batch]

      masks = None
      if len(batch) == 6: # VG
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 8: # COCO
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, triplet_masks = batch
      #elif len(batch) == 9: # COCO
      #  imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, triplet_masks, triplet_contours = batch
      #elif len(batch) == 7:
      #  imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1]
	
      # Run the model as it has been run during training
      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
      # imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
      #imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triple_boxes_pred, triple_boxes_gt, triplet_masks_pred, triplet_contours_pred = model_out
      imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triple_boxes_pred, triple_boxes_gt, triplet_masks_pred = model_out

      num_batch_samples = imgs.size(0)
      num_samples += num_batch_samples
      if num_samples >= args.num_val_samples:
        break

      super_boxes = []

      # open file to record all triplets, per image, in a batch
      file_path = os.path.join(img_dir, 'all_batch_triplets.txt')
      f = open(file_path,'w')
      ### embedding stuff below here ####
      for i in range(0,num_batch_samples):
        print('Processing image', i+1, 'of batch size', args.batch_size)
        f.write('---------- image ' + str(i) + '----------\n')

        # from batch: objs, triples, triple_to_img, objs_to_img (need indices in that to select to tie triplets to image)
        # from model: obj_embed, pred_embed

        # find all triple indices for specific image
        # all triples for image i
        # TODO: clean up code so it is numpy() equivalent in all places
        tr_index = np.where(triple_to_img.cpu().numpy() == i)
        tr_img = triples.cpu().numpy()[tr_index, :]
        tr_img = np.squeeze(tr_img, axis=0)
        # 8 point triple boxes
        np_triple_boxes_gt = np.array(triple_boxes_gt).astype(float)
        tr_img_boxes = np_triple_boxes_gt[tr_index]
        assert len(tr_img) == len(tr_img_boxes)

        # vocab['object_idx_to_name'], vocab['pred_idx_to_name']
        # s,o: indices for "objs" array (yields 'object_idx' for 'object_idx_to_name')
        # p: use this value as is (yields 'pred_idx' for 'pred_idx_to_name')
        s, p, o = np.squeeze(np.split(tr_img, 3, axis=1))

        # iterate over all triplets in image to form (subject, predicat, object) tuples
        relationship_data = []
        num_triples = len(tr_img)

	# need to iterate over all triples due to information that needs to be extracted per triple
        for n in range(0, num_triples):
          # tuple = (objs[obj_index], p, objs[subj_index])
          subj_index = s[n]
          subj = np.array(model.vocab['object_idx_to_name'])[objs[subj_index]]
          pred = np.array(model.vocab['pred_idx_to_name'])[p[n]]
          obj_index = o[n]
          obj = np.array(model.vocab['object_idx_to_name'])[objs[obj_index]]
          triplet = tuple([subj, pred, obj])
          relationship_data += [tuple([subj, pred, obj])]
          print(tuple([subj, pred, obj]))
          #print('--------------------')
          f.write('(' + db_utils.tuple_to_string(tuple([subj, pred, obj])) + ')\n')

          # GT bounding boxes: (x0, y0, x1, y1) format, in a [0, 1] coordinate system
          # (from "boxes" (one for each object in "objs") using subj_index and obj_index)
          subj_bbox = tr_img_boxes[n,0:5] 
          obj_bbox = tr_img_boxes[n, 4:8] 
          print(tuple([subj, pred, obj]), subj_bbox, obj_bbox)

          # SG GCNN embeddings to be used for search (nth triplet corresponds to nth embedding)
          #subj_embed = obj_embeddings[subj_index].numpy().tolist()
          #pred_embed = pred_embeddings[n].numpy().tolist()
          #obj_embed = obj_embeddings[obj_index].numpy().tolist()
          subj_embed = obj_embeddings[subj_index].cpu().numpy().tolist()
          pred_embed = pred_embeddings[n].cpu().numpy().tolist()
          obj_embed = obj_embeddings[obj_index].cpu().numpy().tolist()
          pooled_embed = subj_embed + pred_embed + obj_embed

          # add relationship to database
          relationship = dict()
          relationship['subject'] = subj
          relationship['predicate'] = pred
          relationship['object'] = obj
          relationship['subject_bbox'] = subj_bbox.tolist() #JSON can't serialize np.array()
          relationship['object_bbox'] = obj_bbox.tolist()

          # get super box
          #min_x = np.min([tr_img_boxes[n][0], tr_img_boxes[n][4]])
          #min_y = np.min([tr_img_boxes[n][1], tr_img_boxes[n][5]])
          #max_x = np.max([tr_img_boxes[n][2], tr_img_boxes[n][6]])
          #max_y = np.max([tr_img_boxes[n][3], tr_img_boxes[n][7]])
          min_x = np.min([subj_bbox[0], obj_bbox[0]])
          min_y = np.min([subj_bbox[1], obj_bbox[1]])
          max_x = np.max([subj_bbox[2], obj_bbox[2]])
          max_y = np.max([subj_bbox[3], obj_bbox[3]])
          #print([min_x, min_y, max_x, max_y])
          #print([_min_x, _min_y, _max_x, _max_y])
          relationship['super_bbox'] = [min_x, min_y, max_x, max_y]
          super_boxes += [relationship['super_bbox']]
          #relationship['subject_embed'] = subj_embed
          #relationship['predicate_embed'] = pred_embed
          #relationship['object_embed'] = obj_embed
          relationship['embed'] = pooled_embed

          if triplet not in triplet_db:
            triplet_db[db_utils.tuple_to_string(triplet)] = [relationship]
          elif triplet in triplet_db:
            triplet_db[db_utils.tuple_to_string(triplet)] += [relationship]
          #pprint.pprint(triplet_db)
          #pdb.set_trace()

        print('---------------------------------')
        #pprint.pprint(relationship_data)
        #pprint.pprint(triplet_db)  # printed per image iteration
        print('------- end of processing for image --------------------------')

      ####### process batch images by visualizing triplets on all #########
      f.close()
      # measure IoU as a basic metric for bbox prediction
      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)

      # detach
      imgs = imgs.detach()
      triplet_masks = triplet_masks.detach()
      triplet_masks_pred = triplet_masks_pred.detach()
      boxes_pred = boxes_pred.detach()
      
      # deprocess (normalize) images
      samples = {}
      samples['gt_imgs'] = imgs

      for k, v in samples.items():
        samples[k] = imagenet_deprocess_batch(v)

      # GT images
      np_imgs = [gt.cpu().numpy().transpose(1,2,0) for gt in imgs]
      np_triplet_masks = [mask.cpu().numpy() for mask in triplet_masks]
      np_triplet_masks_pred = [mask.cpu().numpy() for mask in triplet_masks_pred]
      # predicted images
      #np_pred_imgs = [p.cpu().numpy().transpose(1,2,0) for p in imgs_pred]

      # visualize predicted boxes/images
      # (output image is always 64x64 based upon how current model is trained)
      pred_overlaid_images = vis.overlay_boxes(np_imgs, model.vocab, objs_vec, boxes_pred, obj_to_img, W=256, H=256)
      # visualize predicted boxes/images
      
      # (output image is always 64x64 based upon how current model is trained)
      #pred_overlaid_masks = vis.overlay_boxes(np_masks_pred, model.vocab, objs_vec, boxes_pred, obj_to_img, W=256, H=256)

     # visualize GT boxes/images
      #overlaid_images = vis.overlay_boxes(np_imgs, model.vocab, objs_vec, boxes, obj_to_img, W=64, H=64)
      overlaid_images = vis.overlay_boxes(np_imgs, model.vocab, objs_vec, boxes, obj_to_img, W=256, H=256)

      # triples to image
      #pdb.set_trace()
      # visualize suberboxes with object boxes underneath
      norm_overlaid_images = [i/255.0 for i in overlaid_images]
      sb_overlaid_images = vis.overlay_boxes(norm_overlaid_images, model.vocab, objs_vec, torch.tensor(super_boxes), triple_to_img, W=256, H=256, drawText=False, drawSuperbox=True)

      import matplotlib.pyplot as plt
      print("---- saving first GT image of batch -----")
      img_gt = np_imgs[0]
      imwrite('./test_GT_img_coco.png', img_gt)
      #plt.imshow(img_gt)  # can visualize [0-1] or [0-255] color scaling
      #plt.show()

      print("---- saving first predicted triplet mask of batch -----")
      gt_mask_np = np_triplet_masks[1]
      plt.imshow(gt_mask_np)
      plt.show()
      pred_mask_np = np_triplet_masks_pred[1]
      #imwrite('./test_pred_overlay_mask_coco.png', img_np)
      plt.imshow(pred_mask_np)
      plt.show()

      print("---- saving first overlay image of batch -----")
      imwrite('./test_overlay_img_coco.png', overlaid_images[0])
      #plt.imshow(overlaid_images[0])
      #plt.show()

      print("---- saving first superbox overlay image of batch -----")
      imwrite('./test_sb_overlay_img_coco.png', sb_overlaid_images[0])
      #plt.imshow(sb_overlaid_images[0])
      #plt.show()

      print("---- saving batch images -----")
      if write_images:
        t = 0
        for gt_img, pred_overlaid_img, overlaid_img, sb_overlaid_img in zip(np_imgs, pred_overlaid_images, overlaid_images, sb_overlaid_images):
          img_path = os.path.join(img_dir, '%06d_gt_img.png' % t)
          imwrite(img_path, gt_img)

          img_path = os.path.join(img_dir, '%06d_pred_bbox.png' % t)
          imwrite(img_path, pred_overlaid_img)

          img_path = os.path.join(img_dir, '%06d_gt_bbox_img.png' % t)
          imwrite(img_path, overlaid_img)

          img_path = os.path.join(img_dir, '%06d_gt_superbox_img.png' % t)
          imwrite(img_path, sb_overlaid_img)
          t += 1

    #pdb.set_trace()
    # write database to JSON file
    db_utils.write_to_JSON(triplet_db, "coco_test_db.json")

    masks_to_store = masks
    if masks_to_store is not None:
      masks_to_store = masks_to_store.data.cpu().clone()

    masks_pred_to_store = masks_pred
    if masks_pred_to_store is not None:
      masks_pred_to_store = masks_pred_to_store.data.cpu().clone()

  #batch_data = {
  #  'objs': objs.detach().cpu().clone(),
  #  'boxes_gt': boxes.detach().cpu().clone(),
  #  'masks_gt': masks_to_store,
  #  'triples': triples.detach().cpu().clone(),
  #  'obj_to_img': obj_to_img.detach().cpu().clone(),
  #  'triple_to_img': triple_to_img.detach().cpu().clone(),
  #  'boxes_pred': boxes_pred.detach().cpu().clone(),
  #  'masks_pred': masks_pred_to_store
  #}
  #out = [mean_losses, samples, batch_data, avg_iou]
  #out = [mean_losses, avg_iou]
  out = []

  ####################
  avg_iou = total_iou / total_boxes
  print('average bbox IoU = ', avg_iou.cpu().numpy())
  ###################

  return tuple(out)
  

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
  # for flags added after model trained.
  checkpoint['model_kwargs']['triplet_box_net'] = args.triplet_box_net
  checkpoint['model_kwargs']['triplet_mask_size'] = args.triplet_mask_size
  checkpoint['model_kwargs']['triplet_embedding_size'] = args.triplet_embedding_size
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
    #print('Extracting embeddings from train set:')
    #train_results = check_model(args, t, train_loader, model, log_tag='Train', write_images=False)
    print('Extracting embeddings from val test set:')
    val_results = check_model(args, t, val_loader, model, log_tag='Validation', write_images=False)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
