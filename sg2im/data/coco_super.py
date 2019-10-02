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

import json, os, random, math, pdb
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils

from .utils import imagenet_preprocess, Resize, compute_object_centers, determine_box_relation

COCO_DIR = os.environ.get('COCO_DIR', '/Users/brigitsc/sandbox/sg2im/datasets/coco/')
#COCO_DIR = os.environ.get('COCO_DIR', '/dataset/coco_stuff/')
COCO_TRAIN_DIR = os.path.join(COCO_DIR, 'images/train2017')
COCO_VAL_DIR = os.path.join(COCO_DIR, 'images/val2017')
COCO_TRAIN_INSTANCES = os.path.join(COCO_DIR, 'annotations/instances_train2017.json')
COCO_TRAIN_STUFF_JSON = os.path.join(COCO_DIR, 'annotations/stuff_train2017.json')
COCO_VAL_INSTANCES = os.path.join(COCO_DIR, 'annotations/instances_val2017.json')
COCO_VAL_STUFF_JSON = os.path.join(COCO_DIR, 'annotations/stuff_val2017.json')


def get_train_val(include_other=False, instance_whitelist=None, stuff_whitelist=None,
                  min_object_size=0.02, min_objects_per_image=3, stuff_only=True,
                  max_train=None, max_val=1024, return_vocab=True):

  base_args = dict(include_other=include_other, instance_whitelist=instance_whitelist,
                  stuff_whitelist=stuff_whitelist, min_object_size=min_object_size,
                  min_objects_per_image=min_objects_per_image, stuff_only=stuff_only)

  train_args = dict(image_dir=COCO_TRAIN_DIR,
                    instances_json=COCO_TRAIN_INSTANCES,
                    stuff_json=COCO_TRAIN_STUFF_JSON,
                    max_samples=max_train, **base_args)

  val_args = dict(image_dir=COCO_VAL_DIR,
                    instances_json=COCO_VAL_INSTANCES,
                    stuff_json=COCO_VAL_STUFF_JSON,
                    max_samples=max_val, **base_args)
  print('Building training set...')
  train = CocoSceneGraphDataset(**train_args)
  print('Building validation set...')
  val = CocoSceneGraphDataset(**val_args)
  if not return_vocab:
    return train, val
    
  assert train.vocab == val.vocab
  vocab = json.loads(json.dumps(train.vocab))
  return train, val, vocab


class CocoSceneGraphDataset(Dataset):
  def __init__(self, image_dir, instances_json, stuff_json=None,
               stuff_only=True, image_size=(64, 64), mask_size=16,
               normalize_images=True, max_samples=None,
               include_relationships=True, min_object_size=0.02,
               min_objects_per_image=3, max_objects_per_image=8,
               include_other=False, instance_whitelist=None, stuff_whitelist=None):
    """
    A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
    them to scene graphs on the fly.

    Inputs:
    - image_dir: Path to a directory where images are held
    - instances_json: Path to a JSON file giving COCO annotations
    - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
    - stuff_only: (optional, default True) If True then only iterate over
      images which appear in stuff_json; if False then iterate over all images
      in instances_json.
    - image_size: Size (H, W) at which to load images. Default (64, 64).
    - mask_size: Size M for object segmentation masks; default 16.
    - normalize_image: If True then normalize images by subtracting ImageNet
      mean pixel and dividing by ImageNet std pixel.
    - max_samples: If None use all images. Other wise only use images in the
      range [0, max_samples). Default None.
    - include_relationships: If True then include spatial relationships; if
      False then only include the trivial __in_image__ relationship.
    - min_object_size: Ignore objects whose bounding box takes up less than
      this fraction of the image.
    - min_objects_per_image: Ignore images which have fewer than this many
      object annotations.
    - max_objects_per_image: Ignore images which have more than this many
      object annotations.
    - include_other: If True, include COCO-Stuff annotations which have category
      "other". Default is False, because I found that these were really noisy
      and pretty much impossible for the system to model.
    - instance_whitelist: None means use all instance categories. Otherwise a
      list giving a whitelist of instance category names to use.
    - stuff_whitelist: None means use all stuff categories. Otherwise a list
      giving a whitelist of stuff category names to use.
    """
    super(Dataset, self).__init__()

    if stuff_only and stuff_json is None:
      print('WARNING: Got stuff_only=True but stuff_json=None.')
      print('Falling back to stuff_only=False.')

    self.image_dir = image_dir
    self.mask_size = mask_size
    self.max_samples = max_samples
    self.normalize_images = normalize_images
    self.include_relationships = include_relationships
    self.set_image_size(image_size)

    with open(instances_json, 'r') as f:
      instances_data = json.load(f)
    
    stuff_data = None
    if stuff_json is not None and stuff_json != '':
      with open(stuff_json, 'r') as f:
        stuff_data = json.load(f)

    self.image_ids = []
    self.image_id_to_filename = {}
    self.image_id_to_size = {}
    for image_data in instances_data['images']:
      image_id = image_data['id']
      filename = image_data['file_name']
      width = image_data['width']
      height = image_data['height']
      self.image_ids.append(image_id)
      self.image_id_to_filename[image_id] = filename
      self.image_id_to_size[image_id] = (width, height)
    
    self.vocab = {
      'object_name_to_idx': {},
      'pred_name_to_idx': {},
    }
    object_idx_to_name = {}
    all_instance_categories = []
    for category_data in instances_data['categories']:
      category_id = category_data['id']
      category_name = category_data['name']
      all_instance_categories.append(category_name)
      object_idx_to_name[category_id] = category_name
      self.vocab['object_name_to_idx'][category_name] = category_id
    all_stuff_categories = []
    if stuff_data:
      for category_data in stuff_data['categories']:
        category_name = category_data['name']
        category_id = category_data['id']
        all_stuff_categories.append(category_name)
        object_idx_to_name[category_id] = category_name
        self.vocab['object_name_to_idx'][category_name] = category_id

    if instance_whitelist is None:
      instance_whitelist = all_instance_categories
    if stuff_whitelist is None:
      stuff_whitelist = all_stuff_categories
    category_whitelist = set(instance_whitelist) | set(stuff_whitelist)


    self.instance_whitelist = instance_whitelist

    # Add object data from instances
    self.image_id_to_objects = defaultdict(list)
    for object_data in instances_data['annotations']:
      image_id = object_data['image_id']
      _, _, w, h = object_data['bbox']
      W, H = self.image_id_to_size[image_id]
      box_area = (w * h) / (W * H)
      box_ok = box_area > min_object_size
      object_name = object_idx_to_name[object_data['category_id']]
      category_ok = object_name in category_whitelist
      other_ok = object_name != 'other' or include_other
      if box_ok and category_ok and other_ok:
        self.image_id_to_objects[image_id].append(object_data)

    # Add object data from stuff
    if stuff_data:
      image_ids_with_stuff = set()
      for object_data in stuff_data['annotations']:
        image_id = object_data['image_id']
        image_ids_with_stuff.add(image_id)
        _, _, w, h = object_data['bbox']
        W, H = self.image_id_to_size[image_id]
        box_area = (w * h) / (W * H)
        box_ok = box_area > min_object_size
        object_name = object_idx_to_name[object_data['category_id']]
        category_ok = object_name in category_whitelist
        other_ok = object_name != 'other' or include_other
        if box_ok and category_ok and other_ok:
          self.image_id_to_objects[image_id].append(object_data)
      if stuff_only:
        new_image_ids = []
        for image_id in self.image_ids:
          if image_id in image_ids_with_stuff:
            new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        all_image_ids = set(self.image_id_to_filename.keys())
        image_ids_to_remove = all_image_ids - image_ids_with_stuff
        for image_id in image_ids_to_remove:
          self.image_id_to_filename.pop(image_id, None)
          self.image_id_to_size.pop(image_id, None)
          self.image_id_to_objects.pop(image_id, None)

    # COCO category labels start at 1, so use 0 for __image__
    self.vocab['object_name_to_idx']['__image__'] = 0

    # Build object_idx_to_name
    name_to_idx = self.vocab['object_name_to_idx']
    assert len(name_to_idx) == len(set(name_to_idx.values()))
    max_object_idx = max(name_to_idx.values())
    idx_to_name = ['NONE'] * (1 + max_object_idx)
    for name, idx in self.vocab['object_name_to_idx'].items():
      idx_to_name[idx] = name
    self.vocab['object_idx_to_name'] = idx_to_name

    # Prune images that have too few or too many objects
    new_image_ids = []
    total_objs = 0
    for image_id in self.image_ids:
      num_objs = len(self.image_id_to_objects[image_id])
      total_objs += num_objs
      if min_objects_per_image <= num_objs <= max_objects_per_image:
        new_image_ids.append(image_id)
    self.image_ids = new_image_ids

    self.vocab['pred_idx_to_name'] = [
      '__in_image__',
      'left of',
      'right of',
      'above',
      'below',
      'inside',
      'surrounding',
      'behind',
      'infront of'
    ]
    self.vocab['pred_name_to_idx'] = {}
    for idx, name in enumerate(self.vocab['pred_idx_to_name']):
      self.vocab['pred_name_to_idx'][name] = idx
    #print(len(self.vocab['pred_idx_to_name']))  


  def set_image_size(self, image_size):
    print('called set_image_size', image_size)
    transform = [Resize(image_size), T.ToTensor()]
    if self.normalize_images:
      transform.append(imagenet_preprocess())
    self.transform = T.Compose(transform)
    self.image_size = image_size

  def total_objects(self):
    total_objs = 0
    for i, image_id in enumerate(self.image_ids):
      if self.max_samples and i >= self.max_samples:
        break
      num_objs = len(self.image_id_to_objects[image_id])
      total_objs += num_objs
    return total_objs

  def __len__(self):
    if self.max_samples is None:
      return len(self.image_ids)
    return min(len(self.image_ids), self.max_samples)

  def __getitem__(self, index):
    """
    Get the pixels of an image, and a random synthetic scene graph for that
    image constructed on-the-fly from its COCO object annotations. We assume
    that the image will have height H, width W, C channels; there will be O
    object annotations, each of which will have both a bounding box and a
    segmentation mask of shape (M, M). There will be T triples in the scene
    graph.

    Returns a tuple of:
    - image: FloatTensor of shape (C, H, W)
    - objs: LongTensor of shape (O,)
    - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
      (x0, y0, x1, y1) format, in a [0, 1] coordinate system
    - masks: LongTensor of shape (O, M, M) giving segmentation masks for
      objects, where 0 is background and 1 is object.
    - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
      means that (objs[i], p, objs[j]) is a triple.
    """
    image_id = self.image_ids[index]
    abs_image = [] 
    filename = self.image_id_to_filename[image_id]
    image_path = os.path.join(self.image_dir, filename)
    with open(image_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        # unscaled image
        abs_image = np.array(image)
        WW, HH = image.size
        image = self.transform(image.convert('RGB'))

    H, W = self.image_size
    objs, boxes, masks = [], [], []
    # absolute masks (unscaled)
    # can later convert masks into self.image_size 
    # scale if scaling factor is known (s = self.image_size/HH, etc)
    abs_boxes, abs_masks = [], []

    # scale 
    for object_data in self.image_id_to_objects[image_id]:
      objs.append(object_data['category_id'])
      x, y, w, h = object_data['bbox']
      x0 = x / WW
      y0 = y / HH
      x1 = (x + w) / WW
      y1 = (y + h) / HH
      boxes.append(torch.FloatTensor([x0, y0, x1, y1]))
      # unscaled boxes in true image coords  (e.g side len = 4 => coords [0,3]
      abs_boxes.append(torch.FloatTensor([x, y, x + w - 1, y + h - 1]))
 
      print('box size h, w: ' , np.round([h, w]).astype(int))
      print(np.round([x, y, x + w - 1, y + h - 1])) 

      # This will give a numpy array of shape (HH, WW)
      mask = seg_to_mask(object_data['segmentation'], WW, HH)

      # Crop the mask according to the bounding box, being careful to
      # ensure that we don't crop a zero-area region
      mx0, mx1 = int(round(x)), int(round(x + w))
      my0, my1 = int(round(y)), int(round(y + h))
      mx1 = max(mx0 + 1, mx1)
      my1 = max(my0 + 1, my1)
      mask = mask[my0:my1, mx0:mx1]
      # unscaled mask
      abs_mask = mask
      print('mask size h/w: ', abs_mask.shape)

      # masks resized to 64x64
      mask = imresize(255.0 * mask, (self.mask_size, self.mask_size),
                      mode='constant', anti_aliasing=True)
      mask = torch.from_numpy((mask > 128).astype(np.int64))
      # unscaled/resized mask 
      abs_mask = torch.from_numpy((abs_mask > 0).astype(np.int64))

      masks.append(mask)
      abs_masks.append(abs_mask)

    # Add dummy __image__ object
    objs.append(self.vocab['object_name_to_idx']['__image__'])
    boxes.append(torch.FloatTensor([0, 0, 1, 1]))
    masks.append(torch.ones(self.mask_size, self.mask_size).long())
    # unscaled box/mask for singleton case
    abs_boxes.append(torch.FloatTensor([0, 0, WW, HH])) # is thiscordering right?
    abs_masks.append(torch.ones(HH, WW).long())

    objs = torch.LongTensor(objs)
    # merge a list of Tensors into one Tensor
    boxes = torch.stack(boxes, dim=0)
    masks = torch.stack(masks, dim=0)
    abs_boxes = torch.stack(abs_boxes, dim=0)

    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Compute centers of all objects
    obj_centers = []
    _, MH, MW = masks.size()
    for i, obj_idx in enumerate(objs):
      x0, y0, x1, y1 = boxes[i]
      mask = (masks[i] == 1)
      xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
      ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
      if mask.sum() == 0:
        mean_x = 0.5 * (x0 + x1)
        mean_y = 0.5 * (y0 + y1)
      else:
        mean_x = xs[mask].mean()
        mean_y = ys[mask].mean()
      obj_centers.append([mean_x, mean_y])
    obj_centers = torch.FloatTensor(obj_centers)

    # Add triples
    triples = []
    num_objs = objs.size(0)
    __image__ = self.vocab['object_name_to_idx']['__image__']
    real_objs = []
    if num_objs > 1:
      real_objs = (objs != __image__).nonzero().squeeze(1)
    for cur in real_objs:
      choices = [obj for obj in real_objs if obj != cur]
      if len(choices) == 0 or not self.include_relationships:
        break
      other = random.choice(choices)
      if random.random() > 0.5:
        s, o = cur, other
      else:
        s, o = other, cur

      # Check for inside / surrounding
      sx0, sy0, sx1, sy1 = boxes[s]
      ox0, oy0, ox1, oy1 = boxes[o]
      d = obj_centers[s] - obj_centers[o]
      theta = math.atan2(d[1], d[0])

      if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
        p = 'surrounding'
      elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
        p = 'inside'
      elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
        p = 'left of'
      elif -3 * math.pi / 4 <= theta < -math.pi / 4:
        p = 'above'
      elif -math.pi / 4 <= theta < math.pi / 4:
        p = 'right of'
      elif math.pi / 4 <= theta < 3 * math.pi / 4:
        p = 'below'
      # add heuristics here
      p = self.vocab['pred_name_to_idx'][p]
      triples.append([s, p, o])

      #########################################################
      # front_back_flag = False
      # s_obj_type = self.vocab['object_idx_to_name'][s]  
      # o_obj_type = self.vocab['object_idx_to_name'][o]  
      # if s_obj_type in self.instance_whitelist and o_obj_type in self.instance_whitelist: 
      #   if sy1 > oy1 and oy1 > sy0 and ox1 > sx0 and ox0 < sx1:
      #     q = 'infront of'
      #     front_back_flag = True
      #   elif sy1 < oy1 and oy1 < sy0 and ox1 < sx0 and ox0 > sx1:
      #     q = 'behind'
      #     front_back_flag = True
        
      # if front_back_flag:
      #   q = self.vocab['pred_name_to_idx'][q]
      #   triples.append([s, q, o])
      #########################################################

    # Add __in_image__ triples
    O = objs.size(0)
    in_image = self.vocab['pred_name_to_idx']['__in_image__']
    for i in range(O - 1):
      triples.append([i, in_image, O - 1])
  
    triples = torch.LongTensor(triples)

    # get bounding boxes for triples
    num_triples = triples.size(0)
    s, p, o = triples.chunk(3, dim=1)
    s_boxes, o_boxes = abs_boxes[s], abs_boxes[o]
    triplet_boxes = torch.cat([torch.squeeze(s_boxes), torch.squeeze(o_boxes)], dim=1)
    triplet_masks = []
    superboxes = []

    print(triples) 
    #pdb.set_trace()
   
    # masks will all be resized to fixed size (32x32) 
    for n in range(0, num_triples):
      s_idx, o_idx = s[n], o[n] # index into object-related arrays
      s_mask = abs_masks[s_idx] # unscaled subject mask 
      o_mask = abs_masks[o_idx] # unscaled object mask 
      # find dimensions of encapsulating triplet superbox 
      min_x = np.min([triplet_boxes[n][0], triplet_boxes[n][4]])
      min_y = np.min([triplet_boxes[n][1], triplet_boxes[n][5]])
      max_x = np.max([triplet_boxes[n][2], triplet_boxes[n][6]])
      max_y = np.max([triplet_boxes[n][3], triplet_boxes[n][7]]) 
      superboxes.append([min_x, min_y, max_x, max_y])
      print('----------------------------------')
      print(superboxes[n])
      min_x, min_y = int(round(min_x)), int(round(min_y))
      max_x, max_y = int(round(max_x)), int(round(max_y))
      print([min_x, min_y, max_x, max_y])
      h = max_y - min_y + 1    
      w = max_x - min_x + 1
      print('[', 0, 0, w-1, h-1, ']')
      # create empty mask the size of superbox
      triplet_mask = torch.zeros(h, w)
      # superbox shift offset
      dx, dy = min_x, min_y
      print('dx, dy: ', dx, dy)
      
      bbs = np.array(np.round(triplet_boxes[n])).astype(int) # indices must be integers 
      print("subj:", bbs[:4], "[", bbs[0] - dx, bbs[1] - dy, bbs[2] - dx, bbs[3] - dy, "]") # 
      print("obj:", bbs[4:], "[", bbs[4] - dx, bbs[5] - dy, bbs[6] - dx, bbs[7] - dy, "]") # 
      # subject mask
      x0, x1 = bbs[0] - dx, (bbs[2] - dx) + 1
      y0, y1 = bbs[1] - dy, (bbs[3] - dy) + 1
      triplet_mask[y0:y1, x0:x1] = s_mask
      
      # object mask 
      x0, x1 = bbs[4] - dx, (bbs[6] - dx) + 1
      y0, y1 = bbs[5] - dy, (bbs[7] - dy) + 1
      triplet_mask[y0:y1, x0:x1] = o_mask

      print(triples[n])
      print('(', self.vocab['object_idx_to_name'][objs[s[n]]], ',', self.vocab['pred_idx_to_name'][p[n]], ',',  self.vocab['object_idx_to_name'][objs[o[n]]], ')')

      import matplotlib.pyplot as plt
       
      # need to deal with case for singleton triples
      # scaled masks: can be rescaled to self.image_size ref frame
      ### scale to fixed size here
      if(self.vocab['object_idx_to_name'][objs[o[n]]] == '__image__'):
        print('--------------------------------')
        pdb.set_trace()
        plt.imshow(triplet_mask)
        plt.show()
        plt.imshow(triplet_image)
        plt.show()
      
      # if scaling factors are known (e.g. self.image_size/orig_h/w)
      # (scaled patches ideal for DB)?
      triplet_masks.append(triplet_mask) 
      
      # crop image in super box
      #triplet_image = abs_image[:, min_y:max_y + 1, min_x:max_x + 1]
      triplet_image = abs_image[min_y:max_y + 1, min_x:max_x + 1, :]
      
      #pdb.set_trace()
      #plt.imshow(triplet_mask)
      #plt.show()
      #plt.imshow(triplet_image)
      #plt.show()

    print('----- end processing of image --------') 
    # this gets passed to coco_collate
    return image, objs, boxes, masks, triples
    

def seg_to_mask(seg, width=1.0, height=1.0):
  """
  Tiny utility for decoding segmentation masks using the pycocotools API.
  """
  if type(seg) == list:
    rles = mask_utils.frPyObjects(seg, height, width)
    rle = mask_utils.merge(rles)
  elif type(seg['counts']) == list:
    rle = mask_utils.frPyObjects(seg, height, width)
  else:
    rle = seg
  return mask_utils.decode(rle)


def coco_collate_fn(batch):
  """
  Collate function to be used when wrapping CocoSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, C, H, W)
  - objs: LongTensor of shape (O,) giving object categories
  - boxes: FloatTensor of shape (O, 4)
  - masks: FloatTensor of shape (O, M, M)
  - triples: LongTensor of shape (T, 3) giving triples
  - obj_to_img: LongTensor of shape (O,) mapping objects to images
  - triple_to_img: LongTensor of shape (T,) mapping triples to images
  """
  all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []
  obj_offset = 0
  # update anything returned by __get_item__ here
  for i, (img, objs, boxes, masks, triples) in enumerate(batch):
    all_imgs.append(img[None])
    if objs.dim() == 0 or triples.dim() == 0:
      continue
    O, T = objs.size(0), triples.size(0)
    all_objs.append(objs)
    all_boxes.append(boxes)
    all_masks.append(masks)
    triples = triples.clone()
    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset
    all_triples.append(triples)

    all_obj_to_img.append(torch.LongTensor(O).fill_(i))
    all_triple_to_img.append(torch.LongTensor(T).fill_(i))
    obj_offset += O

  all_imgs = torch.cat(all_imgs)
  all_objs = torch.cat(all_objs)
  all_boxes = torch.cat(all_boxes)
  all_masks = torch.cat(all_masks)
  all_triples = torch.cat(all_triples)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  out = (all_imgs, all_objs, all_boxes, all_masks, all_triples,
         all_obj_to_img, all_triple_to_img)
  return out

