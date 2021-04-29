import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sg2ImModel(nn.Module):
  def __init__(self, vocab,
               *args,
               **kwargs):
    super(Sg2ImModel, self).__init__()

    self.vocab = vocab

    # end of init

  def forward(self, objs, triples, obj_to_img=None, spatial_attributes=None,
              boxes_gt=None, masks_gt=None, tr_to_img=None):
    """
    Required Inputs:
    - objs: LongTensor of shape (O,) giving categories for all objects
    - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
      means that there is a triple (objs[s], p, objs[o])

    Optional Inputs:
    - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
      the spatial layout; if not given then use predicted boxes.
    """

    img = boxes_pred = masks_pred = layout = layout_boxes = layout_masks = obj_to_img = sg_context_pred = sg_context_pred_d = rel_scores = triplet_boxes_pred = triplet_boxes_gt = triplet_masks_pred = boxes_pred_info = triplet_superboxes_pred =  obj_scores = pred_mask_gt = pred_mask_scores = context_tr_vecs = input_tr_vecs = obj_class_scores = rel_class_scores = subj_scores = rel_embedding = mask_rel_embedding = pred_ground = None 
  
    # embedding vectors
    obj_vecs  = torch.zeros(len(objs), 128) 
    pred_vecs =  torch.zeros(len(triples),128) 

    return img, boxes_pred, masks_pred, objs, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, rel_scores, obj_vecs, pred_vecs, triplet_boxes_pred, triplet_boxes_gt, triplet_masks_pred, boxes_pred_info, triplet_superboxes_pred, obj_scores, pred_mask_gt, pred_mask_scores, context_tr_vecs, input_tr_vecs, obj_class_scores, rel_class_scores, subj_scores, rel_embedding, mask_rel_embedding, pred_ground 



