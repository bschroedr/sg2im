
import sys
sys.path.append("./") # in order to include sg2im as module
import argparse
import pprint
import os
import json
import math
from collections import defaultdict
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from sg2im.data.vg_ss import VgSceneGraphDataset, vg_collate_fn
from sg2im.model_layout_SS import Sg2ImModel
import pdb

VG_DIR = '/home/brigit/sandbox/sg2im/datasets/vg'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool)

# VG-specific options
parser.add_argument('--vg_dir', default=VG_DIR)
parser.add_argument('--use_split', default='val', choices=['val', 'test'])
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool)
parser.add_argument('--random_seed', default=42, type=int)
parser.add_argument('--no_images', default=False, type=bool)

# Model options
parser.add_argument('--checkpoint', default='sg2im-models/coco64.pt')
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])

def generate_db(args, loader, vocab, model):
  num_samples = 0 
  #triple_data_to_entry = {}
  image_id_to_entries = {}
  e = 0
  img_count = 0
  db = [] 
  # iterate over all batches of images
  with torch.no_grad():
    for batch in loader:
      if torch.cuda.is_available():
        #batch = [tensor.cuda() for tensor in batch]
        b = []
        # accounnt for elements returned by batch loader are not cuda-compatible (e.g. list) (v''g_eval.py)
        for tensor in batch:
          if isinstance(tensor, list):
            b.append(tensor)
          else:
            b.append(tensor.cuda())
        batch = b
      else:
        batch = [tensor for tensor in batch]
     
      print('Batch loading data')

      # get batch
      imgs, objs, boxes, triples, obj_to_img, triple_to_img, num_attrs, attrs, sp_attrs = batch
      #imgs, objs, boxes, triples, obj_to_img, triple_to_img, num_attrs, attrs, urls = batch

      # run model
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, spatial_attributes=sp_attrs, 
      		        masks_gt=None, tr_to_img=triple_to_img)
      imgs_pred, boxes_pred, masks_pred, objs_vec, layout, layout_boxes, layout_masks, obj_to_img, sg_context_pred, sg_context_pred_d, predicate_scores, obj_embeddings, pred_embeddings, triple_boxes_pred, triple_boxes_gt, triplet_masks_pred, boxes_pred_info, triplet_superboxes_pred, obj_scores, pred_mask_gt, pred_mask_scores, context_tr_vecs, input_tr_vecs, obj_class_scores, rel_class_scores, subj_scores, rel_embedding, mask_rel_embedding, pred_ground = model_out

      num_samples += args.batch_size 
      if num_samples > args.num_val_samples:
        break

      for i in range(0, args.batch_size): 
        img_url = '' 
        #img_url = urls[i]
        # temporary
        img_id = img_count
        img_count += 1

        # process all triples (e.g. relationships) in image
        tr_index = np.where(triple_to_img == i)
        tr_img = triples[tr_index]
        num_triples = len(tr_img)

        # s,o: indices for "objs" array (yields 'object_idx' for 'object_idx_to_name')
        # p: use this value as is (yields 'pred_idx' for 'pred_idx_to_name')
        #s, p, o, rel_ids = np.split(tr_img, 4, axis=1)
        s, p, o = np.split(tr_img, 3, axis=1)

        #pdb.set_trace()
        for n in range(0, num_triples):
          subj_index = s[n]
          subj = np.array(vocab['object_idx_to_name'])[objs[subj_index]]
          pred = np.array(vocab['pred_idx_to_name'])[p[n]]
          obj_index = o[n]
          obj = np.array(vocab['object_idx_to_name'])[objs[obj_index]]
          if obj == '__image__': # singleton objects
            continue
          rel_id = 0 
          #rel_id = rel_ids[n].item()
          
          subj_box = boxes[subj_index][0].tolist()
          obj_box = boxes[obj_index][0].tolist()

          subj_embed = obj_embeddings[subj_index].cpu().numpy() #.tolist()
          pred_embed = pred_embeddings[n].cpu().numpy() #.tolist()
          obj_embed = obj_embeddings[obj_index].cpu().numpy() #.tolist()
          pooled_embed = np.mean([subj_embed,pred_embed,obj_embed], axis=0).tolist()
          
          relationship = {}
          relationship['metadata'] = {'image_url': img_url, 'vg_scene_id': img_id, 'vg_relationship_id': rel_id}
          relationship['data'] = {'s_box': subj_box, 'o_box': obj_box, 'subject': subj, 'predicate': pred, 'object': obj, 'embed': pooled_embed}
          db.append(relationship)
          #pprint.pprint(relationship)
          # keep track of which relationship and image belong to each db entry
          if img_id not in image_id_to_entries:
            image_id_to_entries[img_id] = [e]
          elif img_id in image_id_to_entries:
            image_id_to_entries[img_id] += [e]
          e += 1
          print('--- image {0} ---'.format(img_id))
  print('Database has', len(db), ' visual relationships.')
  return db, image_id_to_entries

def generate_queries(args, db, image_id_to_entries, num_queries=1000):
  queries = [] 
  #num_entries = len(db)
  # need to associate with number of queries
  #img_ids = range(0,100) # permute randomly based upon num_queries
  #entries_to_delete = []
  #for i in img_ids: 
  #  query_id = image_id_to_entries[i][0] # query is first triple in set
  #  queries.append(db[query_id])
  #  entries_to_delete += image_id_to_entries[i]

  # delete queries from db
  #for e in entries_to_delete:
  #  db[e] = ['void']
  #db = [x for x in db if x != ['void']]

  return db, queries

def build_vg_dsets(args):
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  #if args.no_images is True:
  #  img_dir = None
  #else:
  #  img_dir = args.image_dir
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.val_h5,
    'image_dir': args.image_dir, # None in generate_docs_db 
    'max_samples': args.num_val_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
    'seed': args.random_seed
  }

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  vg_dset = VgSceneGraphDataset(**dset_kwargs)

  return vocab, vg_dset

def build_loaders(args):
  if args.dataset == 'vg':
    vocab, vg_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': False,
    'collate_fn': collate_fn,
  }

  val_loader = DataLoader(vg_dset, **loader_kwargs)
  return vocab, val_loader

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
  assert os.path.isfile(args.checkpoint)
  checkpoint = torch.load(args.checkpoint, map_location=map_location)
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'], strict=False)
  model.eval()
  model.to(device)

  # data loader for VG
  vocab, val_loader = build_loaders(args)
  # create test db 
  db, image_id_to_entries = generate_db(args, val_loader, vocab, model)  
  # generate query db
  db, queries = generate_queries(args, db, image_id_to_entries)
  # write documents and queries
  with open('docs.json', 'w') as json_file:
    json.dump(db, json_file)
  #with open('queries.json', 'w') as json_file:
  #  json.dump(queries, json_file)
  #pprint.pprint(queries)
  print('The End')
  
if __name__ == '__main__':
  args = parser.parse_args()
  # where VG data is located
  args.val_h5 = os.path.join(args.vg_dir, args.use_split + '.h5')
  args.vocab_json = os.path.join(args.vg_dir, 'vocab.json')
  args.image_dir = os.path.join(args.vg_dir, 'images')
  main(args)
