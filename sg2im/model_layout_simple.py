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
  def __init__(self, vocab, image_size=(64, 64), embedding_dim=128,
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
               sp_attributes_dim=35,
               **kwargs):
    super(Sg2ImModel, self).__init__()

    # We used to have some additional arguments: 
    # vec_noise_dim, gconv_mode, box_anchor, decouple_obj_predictions
    if len(kwargs) > 0:
      print('WARNING: Model got unexpected kwargs ', kwargs)

    self.vocab = vocab
    self.image_size = image_size
    self.layout_noise_dim = layout_noise_dim
    self.gcnn_pooling = gcnn_pooling 
    self.triplet_box_net = triplet_box_net 
    self.triplet_mask_size = triplet_mask_size
    self.triplet_embedding_size = triplet_embedding_size
    self.use_bbox_info = use_bbox_info
    self.triplet_superbox_net = triplet_superbox_net
    self.use_masked_sg = use_masked_sg
    self.embedding_dim = embedding_dim
    self.sp_attributes_dim = sp_attributes_dim 
  
    num_objs = len(vocab['object_idx_to_name'])
    num_preds = len(vocab['pred_idx_to_name'])
   
    #self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
    #self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)
  
    #if gconv_num_layers == 0:
    #  self.gconv = nn.Linear(embedding_dim, gconv_dim)
    #elif gconv_num_layers > 0:
    #  gconv_kwargs = {
    #    'input_dim': embedding_dim,
    #    'spatial_attributes_dim': sp_attributes_dim,
    #    'output_dim': gconv_dim,
    #    'hidden_dim': gconv_hidden_dim,
    #    'pooling': gconv_pooling,
    #    'mlp_normalization': mlp_normalization,
    #  }
    #  self.gconv = GraphTripleConv(**gconv_kwargs)

    #self.gconv_net = None
    #if gconv_num_layers > 1:
    #  gconv_kwargs = {
    #    'input_dim': gconv_dim,
    #    'hidden_dim': gconv_hidden_dim,
    #    'pooling': gconv_pooling,
    #    'num_layers': gconv_num_layers - 1,
    #    'mlp_normalization': mlp_normalization,
    #  }
    #  self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

    # end of init

  def forward(self, objs, triples, obj_to_img=None, spatial_attributes=None,
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

    img = boxes_pred = masks_pred = layout = layout_boxes = layout_masks = obj_to_img = sg_context_pred = sg_context_pred_d = rel_scores = triplet_boxes_pred = triplet_boxes_gt = triplet_masks_pred = boxes_pred_info = triplet_superboxes_pred =  obj_scores = pred_mask_gt = pred_mask_scores = context_tr_vecs = input_tr_vecs = obj_class_scores = rel_class_scores = subj_scores = rel_embedding = mask_rel_embedding = pred_ground = None 
  
    # embedding vectors
    obj_vecs  = torch.zeros(len(objs), self.embedding_dim) 
    pred_vecs =  torch.zeros(len(triples), self.embedding_dim) 

    return img, boxes_pred, masks_pred, objs, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, rel_scores, obj_vecs, pred_vecs, triplet_boxes_pred, triplet_boxes_gt, triplet_masks_pred, boxes_pred_info, triplet_superboxes_pred, obj_scores, pred_mask_gt, pred_mask_scores, context_tr_vecs, input_tr_vecs, obj_class_scores, rel_class_scores, subj_scores, rel_embedding, mask_rel_embedding, pred_ground 


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

