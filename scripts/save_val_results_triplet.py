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
import pdb

from sg2im.data.coco_aug import CocoSceneGraphDataset, coco_collate_fn
#from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator, CondGANPatchDiscriminator, CondGANDiscriminator
from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard, relation_score

from sg2im.model_layout import Sg2ImModel
#from sg2im.model import Sg2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager

####
from sg2im.perceptual_loss import FeatureExtractor
from sg2im.data.utils import imagenet_deprocess_batch
from sg2im.data.utils import perc_process_batch
#from sg2im.logger import Logger

from imageio import imwrite


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
parser.add_argument('--random_seed', default=42, type=int)  #For comparative results/debug

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
      batch = [tensor.cuda() for tensor in batch]
      #batch = [tensor for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 8:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, triplet_masks = batch
      #elif len(batch) == 7:
      #  imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1]

      objs = objs.detach()
      triples = triples.detach()
      # Run the model as it has been run during training
      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
      # imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
      #imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triple_boxes_pred, triple_boxes_gt = model_out
      imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triple_boxes_pred, triple_boxes_gt, triplet_masks_pred = model_out
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

#def check_model(args, t, loader, model, logger=None, log_tag='', write_images=False):
def check_model(args, t, loader, model, log_tag='', write_images=False):
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor
  #float_dtype = torch.FloatTensor
  #long_dtype = torch.LongTensor
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
  with torch.no_grad():
    for batch in loader:
      batch = [tensor.cuda() for tensor in batch]
      #batch = [tensor for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 8:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, triplet_masks = batch
      #elif len(batch) == 7:
      #  imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1]

      # Run the model as it has been run during training
      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
      # imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
      #imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triple_boxes_pred, triple_boxes_gt = model_out
      imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triple_boxes_pred, triple_boxes_gt, triplet_masks_pred = model_out

      skip_pixel_loss = True
      skip_perceptual_loss = True
      total_loss, losses =  calculate_model_losses(
                                args, skip_pixel_loss, model, imgs, imgs_pred,
                                boxes, boxes_pred, masks, masks_pred,
                                predicates, predicate_scores,
                                skip_perceptual_loss)

      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)

      for loss_name, loss_val in losses.items():
        all_losses[loss_name].append(loss_val)
      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break

      ###################################################
      # can't use this since image generation is turned off.
      samples = {}
      samples['gt_img'] = imgs

      #model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
      #samples['gt_box_gt_mask'] = model_out[0]

      #model_out = model(objs, triples, obj_to_img, boxes_gt=boxes)
      #samples['gt_box_pred_mask'] = model_out[0]

      #model_out = model(objs, triples, obj_to_img)
      #samples['pred_box_pred_mask'] = model_out[0]

      #for k, v in samples.items():
      #  samples[k] = imagenet_deprocess_batch(v)

      #if write_images:
        #3. Log ground truth and predicted images
      #  with torch.no_grad():
      #    p_imgs = samples['gt_box_gt_mask'].detach()
      #    gt_imgs = samples['gt_img'].detach()
      #    p_gbox_pmsk_img = samples['gt_box_pred_mask']
      #    p_test_imgs = samples['pred_box_pred_mask']

      #  np_gt_imgs = [gt.cpu().numpy().transpose(1,2,0) for gt in gt_imgs]
      #  np_pred_imgs = [pred.cpu().numpy().transpose(1,2,0) for pred in p_imgs]
      #  np_gbox_pmsk_imgs = [pred.cpu().numpy().transpose(1,2,0) for pred in p_gbox_pmsk_img]
      #  np_test_pred_imgs = [pred.cpu().numpy().transpose(1,2,0) for pred in p_test_imgs]
      #  np_all_imgs = []


      #  for gt_img, gtb_gtm_img, gtb_pm_img, pred_img in zip(np_gt_imgs, np_pred_imgs, np_gbox_pmsk_imgs, np_test_pred_imgs):
      #    img_path = os.path.join(img_dir, '%06d_gt_img.png' % t)
      #    imwrite(img_path, gt_img)

      #    img_path = os.path.join(img_dir, '%06d_gtb_gtm_img.png' % t)
      #    imwrite(img_path, gtb_gtm_img)

      #    img_path = os.path.join(img_dir, '%06d_gtb_pm_img.png' % t)
      #    imwrite(img_path, gtb_pm_img)

      #    img_path = os.path.join(img_dir, '%06d_pred_img.png' % t)
      #    imwrite(img_path, pred_img)

      #    t=t+1

        # for gt_img, gtb_gtm_img, gtb_pm_img, pred_img in zip(np_gt_imgs, np_pred_imgs, np_gbox_pmsk_imgs, np_test_pred_imgs):
        #   np_all_imgs.append((gt_img * 255.0).astype(np.uint8))
        #   np_all_imgs.append((gtb_gtm_img * 255.0).astype(np.uint8))
        #   np_all_imgs.append((gtb_pm_img * 255.0).astype(np.uint8))
        #   np_all_imgs.append((pred_img * 255.0).astype(np.uint8))

        # logger.image_summary(log_tag, np_all_imgs, t)
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
  #out = [mean_losses, samples, batch_data, avg_iou]
  out = [mean_losses, avg_iou]

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

#  l1_pixel_loss = F.l1_loss(img_pred, img)
#  total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
#                        l1_pixel_weight)

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
    #val_results = check_model(args, t, val_loader, model, logger=logger, log_tag='Validation', write_images=True)
    #val_results = check_model(args, t, val_loader, model, log_tag='Validation', write_images=True)

    rel_score, avg_iou = get_rel_score(args, t, val_loader, model)
    print ('relation score: ', rel_score)
    print ('get_rel_score average iou: ', avg_iou)
    #val_losses, val_avg_iou = val_results
    #print('val iou: ', val_avg_iou)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
