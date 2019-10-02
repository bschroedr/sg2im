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

    num_objs = len(vocab['object_idx_to_name'])
    num_preds = len(vocab['pred_idx_to_name'])
    self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
    self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

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

    box_net_dim = 4
    box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
    self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

    # TODO: add flag for triplet box net
    self.triplet_box_net = None
    # output dimension
    triplet_box_net_dim = 8
    # input dimension is 3*128 for concatenated triplet
    triplet_box_net_layers = [3*gconv_dim, gconv_hidden_dim, triplet_box_net_dim]
    self.triplet_box_net = build_mlp(triplet_box_net_layers, batch_norm=mlp_normalization)

    # triplet embedding network 
    #triplet_embed_dim = 128
    # input dimension is 3*128 for concatenated triplet
    #triplet_embed_layers = [3*gconv_dim, gconv_hidden_dim, triplet_embed_dim]
    # INPUT is triplet_embedding (128)
    #self.triplet_embed_net = build_mlp(triplet_embed_layers, batch_norm=mlp_normalization)

    # object contours net
    self.contour_size = 2*(4 + 4*2) # max # contour pts per object
    self.contour_net = None
    contour_net_layers = [gconv_dim, gconv_hidden_dim, self.contour_size]
    if self.contour_size is not None and self.contour_size > 0:
      self.contour_net = build_mlp(contour_net_layers, batch_norm=mlp_normalization)

    self.mask_net = None
    if mask_size is not None and mask_size > 0:
      self.mask_net = self._build_mask_net(num_objs, gconv_dim+self.contour_size, mask_size)
      #self.mask_net = self._build_mask_net(num_objs, gconv_dim, mask_size)

    # TODO: add flag for triplet mask net
    #self.triplet_mask_size = self.image_size[0]/2
    #self.triplet_mask_net = None
    #self.triplet_contour_size = 2*100 
    #if self.triplet_mask_size is not None and self.triplet_mask_size > 0:
      # input dimsn is 3*128 for concatenated triplet, output dimsn is triplet_mask_size
    #  self.triplet_mask_net = self._build_triplet_mask_net(num_objs, 3*gconv_dim, self.triplet_mask_size)

    # triplet contours net
    #self.triplet_contour_net = None
    #self.triplet_contour_size = 2*100 
    #triplet_contour_net_layers = [3*gconv_dim, gconv_hidden_dim, self.triplet_contour_size]
    #if self.triplet_contour_size is not None and self.triplet_contour_size > 0:
    #  self.triplet_contour_net = build_mlp(triplet_contour_net_layers, batch_norm=mlp_normalization)
    
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
    return nn.Sequential(*layers)

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
              boxes_gt=None, masks_gt=None):
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
    
    if isinstance(self.gconv, nn.Linear):
      obj_vecs = self.gconv(obj_vecs)
    else:
      obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
    if self.gconv_net is not None:
      obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

    boxes_pred = None
    # don't use for contour prediction
    # bounding box prediction
    #boxes_pred = self.box_net(obj_vecs)

    #### predict object contours
    obj_contours_pred = None
    if self.contour_net is not None:
      obj_contours_pred = self.contour_net(obj_vecs) 

    masks_pred = None
    layout_masks = None
    if self.mask_net is not None:
      contour_input = torch.cat([obj_vecs, obj_contours_pred], dim=1)
      mask_scores = self.mask_net(contour_input[:,:,None, None])
      ###mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
      masks_pred = mask_scores.squeeze(1).sigmoid()

    # this only affects training if loss is non-zero
    ##s_boxes, o_boxes = boxes_pred[s], boxes_pred[o]
    ##s_vecs, o_vecs = obj_vecs_orig[s], obj_vecs_orig[o]
    # uses predicted subject/object boxes, original subject/object embedding (input to GCNN)
    ##rel_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, o_vecs], dim=1)
    ##rel_scores = self.rel_aux_net(rel_aux_input)
    rel_scores = None

    # triplet box net
    # idea: using joint obj/pred embedding to predict 8 point bounding box
    #       AND use the INPUT to this function as a _search embedding_
    #       (or a pooled/MLP version of it)
    triplet_boxes_pred = None
    s_vecs_pred, o_vecs_pred = obj_vecs[s], obj_vecs[o] 
    # form triplet embedding vectors
    triplet_input = torch.cat([s_vecs_pred, pred_vecs, o_vecs_pred], dim=1)
    # predict 8 point bounding boxes
    ##triplet_boxes_pred = self.triplet_box_net(triplet_input) 

    # triplet contours for subject/object
    triplet_contour_pred = None
    ##if self.triplet_contour_net is not None:
     ## triplet_contour_pred = self.triplet_contour_net(triplet_input) 
    
    # predict triplet masks using predicted triplet embedding
    triplet_masks_pred = None
    #if self.triplet_mask_net is not None:
      # input dimension must be [h, w, 1, 1]
      # triplet + contours as input
      #triplet_contour_input = torch.cat([s_vecs_pred, pred_vecs, o_vecs_pred, triplet_contour_pred], dim=1)
      #triplet_mask_scores = self.triplet_mask_net(triplet_contour_input[:,:,None, None])
      # triplet as input 
      ##triplet_mask_scores = self.triplet_mask_net(triplet_input[:,:,None, None])
      #triplet_masks_pred = triplet_mask_scores.squeeze(1).sigmoid()
      ##triplet_masks_pred = triplet_mask_scores.squeeze(1)

    H, W = self.image_size
    
    layout_boxes = boxes_pred if boxes_gt is None else boxes_gt
     #### end processing for layout model ####
     
    # compose layout mask (used for CRN, not used in layout model) 
    if masks_pred is None:
      # use only bounding boxes to create layout 
      layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
    else:
      # use bounding boxes + predicted masks to create layout 
      layout_masks = masks_pred
      #layout_masks = masks_pred if masks_gt is None else masks_gt
      layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks, # this is for CRN
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
    return img, boxes_pred, masks_pred, objs, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, rel_scores, obj_vecs, pred_vecs, triplet_boxes_pred, triplet_boxes_gt, triplet_masks_pred, triplet_contour_pred, obj_contours_pred

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

