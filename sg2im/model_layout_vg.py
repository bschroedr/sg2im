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
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sg2im.box_utils as box_utils
from sg2im.graph import GraphTripleConv, GraphTripleConvNet
from sg2im.crn import RefinementNetwork
from sg2im.layout import boxes_to_layout, masks_to_layout, sg_context_to_layout
from sg2im.layers import build_mlp


class Sg2ImModel(nn.Module):
  def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
               gconv_dim=128, gconv_hidden_dim=512,
               gconv_pooling='avg', gconv_num_layers=5,
               refinement_dims=(1024, 512, 256, 128, 64),
               normalization='batch', activation='leakyrelu-0.2',
               mask_size=None, mlp_normalization='none', layout_noise_dim=0,
               sg_context_dim=0, #None, 
               sg_context_dim_d=0, #None, 
               gcnn_pooling='avg',
               triplet_box_net=False,
               triplet_mask_size=0,
               triplet_embedding_size=0,
               use_bbox_info=False,
               triplet_superbox_net=False,
               use_masked_sg=False,
               **kwargs):
    super(Sg2ImModel, self).__init__()

    # We used to have some additional arguments: 
    # vec_noise_dim, gconv_mode, box_anchor, decouple_obj_predictions
    if len(kwargs) > 0:
      print('WARNING: Model got unexpected kwargs ', kwargs)

    self.vocab = vocab
    self.image_size = image_size
    self.layout_noise_dim = layout_noise_dim
    self.sg_context_dim = sg_context_dim 
    self.sg_context_dim_d = sg_context_dim_d 
    self.gcnn_pooling = gcnn_pooling 
    self.triplet_box_net = triplet_box_net 
    self.triplet_mask_size = triplet_mask_size
    self.triplet_embedding_size = triplet_embedding_size
    self.use_bbox_info = use_bbox_info
    self.triplet_superbox_net = triplet_superbox_net
    self.use_masked_sg = use_masked_sg
    # hack to deal with vocabs with differing # of predicates
    self.mask_pred = 46 # vocab['idx_to_pred_name'][46] = 'none'
    #self.mask_pred = vocab['pred_name_to_idx']['none']
    self.embedding_dim = embedding_dim 
  
    num_objs = len(vocab['object_idx_to_name'])
    num_preds = len(vocab['pred_idx_to_name'])
   
    self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
    self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)
  
    # frozen embedding layers 
    self.fr_obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
    self.fr_pred_embeddings = nn.Embedding(num_preds, embedding_dim)
    self.fr_obj_embeddings.requires_grad = False
    self.fr_pred_embeddings.requires_grad = False

    if gconv_num_layers == 0:
      self.gconv = nn.Linear(embedding_dim, gconv_dim)
    elif gconv_num_layers > 0:
      gconv_kwargs = {
        'input_dim': embedding_dim,
        'output_dim': gconv_dim,
        'hidden_dim': gconv_hidden_dim,
        'pooling': gconv_pooling,
        'mlp_normalization': mlp_normalization,
      }
      self.gconv = GraphTripleConv(**gconv_kwargs)

    self.gconv_net = None
    if gconv_num_layers > 1:
      gconv_kwargs = {
        'input_dim': gconv_dim,
        'hidden_dim': gconv_hidden_dim,
        'pooling': gconv_pooling,
        'num_layers': gconv_num_layers - 1,
        'mlp_normalization': mlp_normalization,
      }
      self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

    if self.use_bbox_info:
      box_net_dim = 4 + 1 # augment with addition info abt bbox
    else:
      box_net_dim = 4
    box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
    self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

    # triplet-related nets 
    self.triplet_box_net = None
    self.triplet_embed_net = None
    self.triplet_mask_net = None
    self.triplet_superbox_net = None
    self.pred_ground_net = None

    # output dimension
    triplet_box_net_dim = 8
    if triplet_box_net:
      # input dimension is 3*128 for concatenated triplet
      triplet_box_net_layers = [3*gconv_dim, gconv_hidden_dim, triplet_box_net_dim]
      self.triplet_box_net = build_mlp(triplet_box_net_layers, batch_norm=mlp_normalization)

    # triplet embedding 
    if self.triplet_embedding_size > 0: 
      # input dimn is 3*128 for concatenated triplet, output dimsn is triplet_embed_dim
      triplet_embed_layers = [3*gconv_dim, gconv_hidden_dim, triplet_embedding_size]
      self.triplet_embed_net = build_mlp(triplet_embed_layers, batch_norm=mlp_normalization)

    if self.triplet_mask_size > 0:
      # input dimsn is 3*128 for concatenated triplet, output dimsn is triplet_mask_size
      #self.triplet_mask_net = self._build_mask_net(num_objs, 3*gconv_dim, self.triplet_mask_size)
      self.triplet_mask_net = self._build_triplet_mask_net(num_objs, 3*gconv_dim, self.triplet_mask_size)

    triplet_superbox_net_dim = 4
    if triplet_superbox_net:
      # input dimension is 3*128 for concatenated triplet
      triplet_superbox_net_layers = [3*gconv_dim, gconv_hidden_dim, triplet_superbox_net_dim]
      self.triplet_superbox_net = build_mlp(triplet_superbox_net_layers, batch_norm=mlp_normalization)

    self.mask_net = None
    if mask_size is not None and mask_size > 0:
      self.mask_net = self._build_mask_net(num_objs, gconv_dim, mask_size)

    ###########################
    self.sg_context_net = None
    self.sg_context_net_d = None
    if sg_context_dim is not None and sg_context_dim > 0:
      H, W = self.image_size
      self.sg_context_net = nn.Linear(gconv_dim, sg_context_dim)
      self.sg_context_net_d = nn.Linear(gconv_dim, sg_context_dim_d) 
      # sg_context_net_layers = [gconv_dim, sg_context_dim]
      # sg_context_net_layers = [gconv_dim, sg_context_dim_d]
      # self.sg_context_net = build_mlp(sg_context_net_layers, batch_norm=mlp_normalization)
      # self.sg_context_net_d = build_mlp(sg_context_net_layers, batch_norm=mlp_normalization)
    ####################### 

    rel_aux_layers = [2 * embedding_dim + 8, gconv_hidden_dim, num_preds]
    self.rel_aux_net = build_mlp(rel_aux_layers, batch_norm=mlp_normalization)

    # subject prediction network
    subj_aux_layers = [2 * embedding_dim + 8, gconv_hidden_dim, num_objs]
    self.subj_aux_net = build_mlp(subj_aux_layers, batch_norm=mlp_normalization)
    
    # object prediction network
    obj_aux_layers = [2 * embedding_dim + 8, gconv_hidden_dim, num_objs]
    self.obj_aux_net = build_mlp(obj_aux_layers, batch_norm=mlp_normalization)
   
    # object class prediction network 
    obj_class_aux_layers = [embedding_dim, gconv_hidden_dim, num_objs]
    self.obj_class_aux_net = build_mlp(obj_class_aux_layers, batch_norm=mlp_normalization)

    # relationship embedding network
    self.rel_embed_aux_net = nn.Linear(embedding_dim, embedding_dim) 
    # relationship class prediction network 
    self.rel_class_aux_net = nn.Linear(embedding_dim, num_preds) 

    # predicate mask prediction network
    pred_mask_layers = [2 * embedding_dim, gconv_hidden_dim, num_preds]
    self.pred_mask_net = build_mlp(pred_mask_layers, batch_norm=mlp_normalization)

    pred_ground_net_dim = 4
    # input dimension 128 for relationship
    pred_ground_net_layers = [gconv_dim, gconv_hidden_dim, pred_ground_net_dim]
    self.pred_ground_net = build_mlp(pred_ground_net_layers, batch_norm=mlp_normalization)

    # input dimn is 3*128 for concatenated triplet
    triplet_context_layers = [4*gconv_dim, gconv_hidden_dim, 3*gconv_dim]
    #self.triplet_context_net = nn.Linear(4*gconv_dim, 3*gconv_dim) 
    #triplet_context_layers = [3*gconv_dim, gconv_hidden_dim, 4]
    #self.triplet_context_net = nn.Linear(4*gconv_dim, 3*gconv_dim) 
    self.triplet_context_net = build_mlp(triplet_context_layers, batch_norm=mlp_normalization)

    if sg_context_dim > 0:
      refinement_kwargs = {
      'dims': (gconv_dim + sg_context_dim + layout_noise_dim,) + refinement_dims,
      'normalization': normalization,
      'activation': activation,
    }
    else:
      refinement_kwargs = {
        'dims': (gconv_dim + layout_noise_dim,) + refinement_dims,
        'normalization': normalization,
        'activation': activation,
      }
    self.refinement_net = RefinementNetwork(**refinement_kwargs)

  def _build_mask_net(self, num_objs, dim, mask_size):
    output_dim = 1
    layers, cur_size = [], 1
    while cur_size < mask_size:
      layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
      layers.append(nn.BatchNorm2d(dim))
      layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
      layers.append(nn.ReLU())
      cur_size *= 2
    if cur_size != mask_size:
      raise ValueError('Mask size must be a power of 2')
    layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
    # freeze layers trained on COCO/MTM
    m = nn.Sequential(*layers)
    for param in m.parameters():
      param.requires_grad = False 
    #for name, param in m.named_parameters():
    #  print(name, param.requires_grad)
    #return nn.Sequential(*layers)
    return m 

  def _build_triplet_mask_net(self, num_objs, dim, triplet_mask_size):
    output_dim = 3 # for 3 object classes: subj, obj, background
    layers, cur_size = [], 1
    while cur_size < triplet_mask_size:
      layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
      layers.append(nn.BatchNorm2d(dim))
      layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
      layers.append(nn.ReLU())
      cur_size *= 2
    if cur_size != triplet_mask_size:
      raise ValueError('Mask size must be a power of 2')
    layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
    return nn.Sequential(*layers)

  def forward(self, objs, triples, obj_to_img=None,
              boxes_gt=None, masks_gt=None, tr_to_img=None):
    """
    Required Inputs:
    - objs: LongTensor of shape (O,) giving categories for all objects
    - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
      means that there is a triple (objs[s], p, objs[o])

    Optional Inputs:
    - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
      means that objects[o] is an object in image i. If not given then
      all objects are assumed to belong to the same image.
    - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
      the spatial layout; if not given then use predicted boxes.
    """
    O, T = objs.size(0), triples.size(0)
    s, p, o = triples.chunk(3, dim=1)           # All have shape (T, 1)
    s, p, o = [x.squeeze(1) for x in [s, p, o]] # Now have shape (T,)
    edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)

    if obj_to_img is None:
      obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

    obj_vecs = self.obj_embeddings(objs)  # 'objs' => indices for model.vocab['object_idx_to_name'] 
    obj_vecs_orig = obj_vecs
    pred_vecs = self.pred_embeddings(p)   #  'p' => indices for model.vocab['pred_idx_to_name']
    pred_vecs_orig = pred_vecs

    if isinstance(self.gconv, nn.Linear):
      obj_vecs = self.gconv(obj_vecs)
    else:
      obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
    if self.gconv_net is not None:
      obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

    #### object context vectors ###############
    num_imgs = obj_to_img[obj_to_img.size(0)-1] + 1
    context_obj_vecs = torch.zeros(num_imgs, obj_vecs.size(1), dtype=obj_vecs.dtype, device=obj_vecs.device)
    obj_to_img_exp = obj_to_img.view(-1, 1).expand_as(obj_vecs)
    context_obj_vecs = context_obj_vecs.scatter_add(0, obj_to_img_exp, obj_vecs)
   
    # get object counts
    obj_counts = torch.zeros(num_imgs, dtype=obj_vecs.dtype, device=obj_vecs.device)
    ones = torch.ones(obj_to_img.size(0), dtype=obj_vecs.dtype, device=obj_vecs.device)
    obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
    context_obj_vecs = context_obj_vecs / obj_counts.view(-1, 1) 
    context_obj_vecs = context_obj_vecs[obj_to_img]
    context_obj_vecs = context_obj_vecs[s]
    ####################################

    ####### triplet context vectors ########### 
    #context_tr_vecs = None
    # concatenate triplet vectors
    #triplets = torch.cat([obj_vecs[s], pred_vecs, obj_vecs[o]], dim=1)
    #context_tr_vecs = torch.zeros(num_imgs, triplets.size(1), dtype=obj_vecs.dtype, device=obj_vecs.device)
    # need triplet to image
    #tr_to_img_exp = tr_to_img.view(-1, 1).expand_as(triplets)
    #context_tr_vecs = context_tr_vecs.scatter_add(0, tr_to_img_exp, triplets)
    # get triplet counts
    #tr_counts = torch.zeros(num_imgs, dtype=obj_vecs.dtype, device=obj_vecs.device)
    #ones = torch.ones(triplets.size(0), dtype=obj_vecs.dtype, device=obj_vecs.device)
    #tr_counts = tr_counts.scatter_add(0, tr_to_img, ones)
    #context_tr_vecs = context_tr_vecs/tr_counts.view(-1,1)
    # dimension is (# triplets, 3*input_dim)
    #context_tr_vecs = context_tr_vecs[tr_to_img]
    # get some context! 
    #context_tr_vecs = self.triplet_context_net(context_tr_vecs)
    ###########################################

    ####  mask out some predicates #####
    pred_mask_gt = None
    pred_mask_scores = None
    if self.use_masked_sg:
      perc = torch.FloatTensor([0.50]) # hyperparameter
      num_mask_objs = torch.floor(perc*len(s)).cpu().numpy()[0].astype(int)
      if num_mask_objs < 1:
        num_mask_objs = 1
      mask_idx = torch.randint(0, len(s)-1, (num_mask_objs,))
      #rand_idx = torch.randperm(len(s)-1)
      #mask_idx = rand_idx[:num_mask_objs]
      # GT
      pred_mask_gt = p[mask_idx.long()]  # return
      # set mask idx to masked embedding (e.g. new SG!)
      pred_vecs_copy = pred_vecs_orig
      ##### need to add i=46 None embedding
      pred_vecs_copy[mask_idx.long()] = self.pred_embeddings(torch.tensor([self.mask_pred]).cuda())
   
      # convolve new masked SG 
      if isinstance(self.gconv, nn.Linear):
        mask_obj_vecs = self.gconv(obj_vecs_orig)
      else:
        mask_obj_vecs, mask_pred_vecs = self.gconv(obj_vecs_orig, pred_vecs_copy, edges)
      if self.gconv_net is not None:
        mask_obj_vecs, mask_pred_vecs = self.gconv_net(mask_obj_vecs, mask_pred_vecs, edges)

      # subj/obj obj idx
      s_mask = s[mask_idx.long()]
      o_mask = o[mask_idx.long()]

      subj_vecs_mask = mask_obj_vecs[s_mask]
      obj_vecs_mask = mask_obj_vecs[o_mask]

      # predict masked predicate relationship
      pred_mask_input = torch.cat([subj_vecs_mask, obj_vecs_mask], dim=1)
      pred_mask_scores = self.pred_mask_net(pred_mask_input)
    #####################
 

    # bounding box prediction
    boxes_pred_info = None
    if self.use_bbox_info:
      # bounding box prediction + predicted box info
      boxes_pred_info = self.box_net(obj_vecs)
      boxes_pred = boxes_pred_info[:,0:4] # first 4 entries are bbox coords
    else:
      boxes_pred = self.box_net(obj_vecs) 

    masks_pred = None
    layout_masks = None
    if self.mask_net is not None:
      mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
      masks_pred = mask_scores.squeeze(1).sigmoid()

    # predicted bboxes and embedding vectors 
    s_boxes, o_boxes = boxes_pred[s], boxes_pred[o]
    s_vecs_pred, o_vecs_pred = obj_vecs[s], obj_vecs[o] 
    # input embedding vectors
    s_vecs, o_vecs, p_vecs = obj_vecs_orig[s], obj_vecs_orig[o], pred_vecs_orig
    input_tr_vecs = torch.cat([s_vecs, p_vecs, o_vecs], dim=1)
 
    # VSA (with obj/pred vectors of varying kinds)
    fr_obj_vecs = self.fr_obj_embeddings(objs) 
    fr_pred_vecs = self.fr_pred_embeddings(p)
    fr_s_vecs, fr_o_vecs  = fr_obj_vecs[s], fr_obj_vecs[o]
    mapc_bind = fr_s_vecs * fr_o_vecs * fr_pred_vecs
    # mapc_bind = s_vecs * o_vecs * p_vecs
    #mapc_bind = s_vecs_pred * o_vecs_pred * pred_vecs
    mapc_bind = F.normalize(mapc_bind, p=2, dim=1)

    # uses predicted subject/object boxes, original subject/object embedding (input to GCNN)
    ## use original embedding vectors
    rel_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, o_vecs], dim=1)
    rel_scores = self.rel_aux_net(rel_aux_input)

    # subject prediction
    subj_aux_input = torch.cat([s_boxes, o_boxes, p_vecs, o_vecs], dim=1)
    subj_scores = self.subj_aux_net(subj_aux_input)

    # object prediction
    obj_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, p_vecs], dim=1)
    obj_scores = self.obj_aux_net(obj_aux_input)

    # object class prediction (for output object vectors)
    obj_class_scores = self.obj_class_aux_net(obj_vecs)
 
    # relationship class prediction (for output object vectors)
    # relationship embedding (very small embedding)
    # augment relationship embedding
    use_augmentation = False 
    mask_rel_embedding = None 
    if use_augmentation:
      num_augs = 4
      num_preds = len(p)
      mask_rel_embedding = torch.zeros((num_preds,num_augs+1,self.embedding_dim), dtype=obj_vecs.dtype, device=obj_vecs.device)
      rel_embedding = []
      #pred_vecs_mask = np.zeros((num_preds,num_augs,self.embedding_dim))
      # mask embedding perc_aug of vectors with 0 
      perc_aug = torch.FloatTensor([0.4]) # hyperparameter
      num_mask = torch.floor(perc_aug*len(pred_vecs[0])).cpu().numpy()[0].astype(int)
      #p_ids = []
      for i in range(num_preds):
        pred = pred_vecs[i]
        #p_id = p[i]
        vecs = [pred] 
        #p_ids += [p_id]
        for j in range(num_augs):
          # pick a random set of indices to zero-out
          rand_idx = torch.randperm(len(pred_vecs[0])) # 0-127 range shuffled
          mask_idx = rand_idx[:num_mask]
          pred_mask = pred.detach().clone()
          pred_mask[mask_idx] = 0.0
          vecs += [pred_mask]
        # project masked augmented relationship vectors  
        pred_mask = self.rel_embed_aux_net(torch.stack(vecs)) # output predicate embeddings
        pred_mask = F.normalize(pred_mask, dim=1)
        mask_rel_embedding[i,:,:] = pred_mask
        rel_embedding += [pred_mask[0]]
   
    #rel_embedding = torch.stack(rel_embedding)
    # projection head for supervised contrastive loss  
    rel_embedding = self.rel_embed_aux_net(pred_vecs) # output projected predicate embeddings
    rel_embedding = F.normalize(rel_embedding, dim=1)

    # relationship class prediction on predicates 
    rel_class_scores = self.rel_class_aux_net(pred_vecs) 

    # concatenate triplet vectors
    s_vecs_pred, o_vecs_pred = obj_vecs[s], obj_vecs[o] 
    triplet_input = torch.cat([s_vecs_pred, pred_vecs, o_vecs_pred], dim=1)

    # triplet bounding boxes
    triplet_boxes_pred = None
    if self.triplet_box_net is not None:
      # predict 8 point bounding boxes
      triplet_boxes_pred = self.triplet_box_net(triplet_input)

    # triplet binary masks
    triplet_masks_pred = None
    if self.triplet_mask_net is not None:
      # input dimension must be [h, w, 1, 1]
      triplet_mask_scores = self.triplet_mask_net(triplet_input[:,:,None, None])
      # only used for binary/masks CE loss
      #triplet_masks_pred = triplet_mask_scores.squeeze(1).sigmoid()
      triplet_masks_pred = triplet_mask_scores.squeeze(1)

    # triplet embedding 
    triplet_embed = None
    if self.triplet_embed_net is not None:
      triplet_embed = self.triplet_embed_net(triplet_input)
 
    # triplet superbox
    triplet_superboxes_pred = None
    if self.triplet_superbox_net is not None:
      # predict 2 point superboxes
      triplet_superboxes_pred = self.triplet_superbox_net(triplet_input) # s/p/o (bboxes?) 

    # predicate grounding 
    pred_ground = None
    if self.pred_ground_net is not None:
      # predict 2 point pred grounding
      pred_ground = self.pred_ground_net(pred_vecs) # s/p/o (bboxes?) 

    # triplet context
    triplet_context_input = torch.cat([context_obj_vecs, s_vecs_pred, pred_vecs, o_vecs_pred], dim=1)
    # output dimension is 384
    context_tr_vecs = self.triplet_context_net(triplet_context_input) 


    H, W = self.image_size
    layout_boxes = boxes_pred if boxes_gt is None else boxes_gt

    # compose layout mask 
    if masks_pred is None:
      layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
    else:
      layout_masks = masks_pred if masks_gt is None else masks_gt
      layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
                               obj_to_img, H, W)
    layout_crn = layout
    sg_context_pred = None
    sg_context_pred_d = None 
    if self.sg_context_net is not None:
      N, C, H, W = layout.size()
      context = sg_context_to_layout(obj_vecs, obj_to_img, pooling=self.gcnn_pooling) 
      sg_context_pred_sqz = self.sg_context_net(context)
      
      #### vector to spatial replication
      b = N
      s = self.sg_context_dim 
      # b, s = sg_context_pred_sqz.size()   
      sg_context_pred = sg_context_pred_sqz.view(b, s, 1, 1).expand(b, s, layout.size(2), layout.size(3));
      layout_crn = torch.cat([layout, sg_context_pred], dim=1)

      ## discriminator uses different FC layer than the generator
      sg_context_predd_sqz = self.sg_context_net_d(context)
      s = self.sg_context_dim_d
      sg_context_pred_d = sg_context_predd_sqz.view(b, s, 1, 1).expand(b, s, layout.size(2), layout.size(3));
      
    if self.layout_noise_dim > 0:
      N, C, H, W = layout.size()
      noise_shape = (N, self.layout_noise_dim, H, W)
      layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                 device=layout.device)
      layout_crn = torch.cat([layout_crn, layout_noise], dim=1)

    # layout model only
    #img = self.refinement_net(layout_crn)
    img = None 

    # compose triplet boxes using 'triplets', objs, etc.
    if boxes_gt is not None:
      s_boxes_gt, o_boxes_gt = boxes_gt[s], boxes_gt[o] 
      triplet_boxes_gt = torch.cat([s_boxes_gt, o_boxes_gt], dim=1)    
    else: 
      triplet_boxes_gt = None
    
    #return img, boxes_pred, masks_pred, rel_scores
    return img, boxes_pred, masks_pred, objs, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, rel_scores, obj_vecs, pred_vecs, triplet_boxes_pred, triplet_boxes_gt, triplet_masks_pred, boxes_pred_info, triplet_superboxes_pred, obj_scores, pred_mask_gt, pred_mask_scores, context_tr_vecs, input_tr_vecs, obj_class_scores, rel_class_scores, subj_scores, rel_embedding, mask_rel_embedding, pred_ground #, mapc_bind


  def encode_scene_graphs(self, scene_graphs):
    """
    Encode one or more scene graphs using this model's vocabulary. Inputs to
    this method are scene graphs represented as dictionaries like the following:

    {
      "objects": ["cat", "dog", "sky"],
      "relationships": [
        [0, "next to", 1],
        [0, "beneath", 2],
        [2, "above", 1],
      ]
    }

    This scene graph has three relationshps: cat next to dog, cat beneath sky,
    and sky above dog.

    Inputs:
    - scene_graphs: A dictionary giving a single scene graph, or a list of
      dictionaries giving a sequence of scene graphs.

    Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
    same semantics as self.forward. The returned LongTensors will be on the
    same device as the model parameters.
    """
    if isinstance(scene_graphs, dict):
      # We just got a single scene graph, so promote it to a list
      scene_graphs = [scene_graphs]

    objs, triples, obj_to_img = [], [], []
    obj_offset = 0
    for i, sg in enumerate(scene_graphs):
      # Insert dummy __image__ object and __in_image__ relationships
      sg['objects'].append('__image__')
      image_idx = len(sg['objects']) - 1
      for j in range(image_idx):
        sg['relationships'].append([j, '__in_image__', image_idx])

      for obj in sg['objects']:
        obj_idx = self.vocab['object_name_to_idx'].get(obj, None)
        if obj_idx is None:
          raise ValueError('Object "%s" not in vocab' % obj)
        objs.append(obj_idx)
        obj_to_img.append(i)
      for s, p, o in sg['relationships']:
        pred_idx = self.vocab['pred_name_to_idx'].get(p, None)
        if pred_idx is None:
          raise ValueError('Relationship "%s" not in vocab' % p)
        triples.append([s + obj_offset, pred_idx, o + obj_offset])
      obj_offset += len(sg['objects'])
    device = next(self.parameters()).device
    objs = torch.tensor(objs, dtype=torch.int64, device=device)
    triples = torch.tensor(triples, dtype=torch.int64, device=device)
    obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
    return objs, triples, obj_to_img

  def forward_json(self, scene_graphs):
    """ Convenience method that combines encode_scene_graphs and forward. """
    objs, triples, obj_to_img = self.encode_scene_graphs(scene_graphs)
    return self.forward(objs, triples, obj_to_img)

