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

import PIL
import torch
import torchvision.transforms as T
import math


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
  return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def _rescale(imgs):
  b_min, b_max = imgs.min(), imgs.max()
  return imgs.sub(b_min).div(b_max - b_min)

def _grayscale():
  transforms = [T.ToPILImage(), T.Grayscale(num_output_channels=3), T.ToTensor()]
  return T.Compose(transforms)


def perc_process_batch(imgs, rescale=True, grayscale=False):
  """
  Input:
  - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images
  Output:
  - imgs_de: FloatTensor of shape (N, C, H, W) giving preprocessed images
    for VGG input
  """
  gray = _grayscale()
  normed_imgs =  _rescale(imgs) if rescale else imgs
  if grayscale:
    normed_imgs = [gray(img.cpu()).cuda() for img in normed_imgs]

  preprocess_fn = imagenet_preprocess()
  imgs_proc = [preprocess_fn(img) for img in normed_imgs]
  return torch.stack(imgs_proc, dim=0)


def imagenet_deprocess(rescale_image=True):
  transforms = [
    T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
    T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
  ]
  if rescale_image:
    transforms.append(_rescale)
  return T.Compose(transforms)

def imagenet_deprocess_batch(imgs, rescale=True, gpu=True):
  """
  Input:
  - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images
  Output:
  - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
    in the range [0, 255]
  """
  deprocess_fn = imagenet_deprocess(rescale_image=rescale)
  imgs_de = [deprocess_fn(img) for img in imgs]
  return torch.stack(imgs_de, dim=0)


class Resize(object):
  def __init__(self, size, interp=PIL.Image.BILINEAR):
    if isinstance(size, tuple):
      H, W = size
      self.size = (W, H)
    else:
      self.size = (size, size)
    self.interp = interp

  def __call__(self, img):
    return img.resize(self.size, self.interp)


def unpack_var(v):
  if isinstance(v, torch.autograd.Variable):
    return v.data
  return v


def split_graph_batch(triples, obj_data, obj_to_img, triple_to_img):
  triples = unpack_var(triples)
  obj_data = [unpack_var(o) for o in obj_data]
  obj_to_img = unpack_var(obj_to_img)
  triple_to_img = unpack_var(triple_to_img)

  triples_out = []
  obj_data_out = [[] for _ in obj_data]
  obj_offset = 0
  N = obj_to_img.max() + 1
  for i in range(N):
    o_idxs = (obj_to_img == i).nonzero().view(-1)
    t_idxs = (triple_to_img == i).nonzero().view(-1)

    cur_triples = triples[t_idxs].clone()
    cur_triples[:, 0] -= obj_offset
    cur_triples[:, 2] -= obj_offset
    triples_out.append(cur_triples)

    for j, o_data in enumerate(obj_data):
      cur_o_data = None
      if o_data is not None:
        cur_o_data = o_data[o_idxs]
      obj_data_out[j].append(cur_o_data)

    obj_offset += o_idxs.size(0)

  return triples_out, obj_data_out


def compute_object_centers(boxes, masks):
  _, MH, MW = masks.size()
  obj_centers = []
  for i, box in enumerate(boxes):
    x0, y0, x1, y1 = box
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

  return torch.Tensor(obj_centers).to(boxes)

      
def determine_box_relation(box_a, box_b, theta, vocab=None):
  sx0, sy0, sx1, sy1 = box_a
  ox0, oy0, ox1, oy1 = box_b
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
  if vocab is not None:
    return vocab['pred_name_to_idx'][p]
  return p
