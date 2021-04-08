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

import tempfile, os
import torch
import numpy as np
import matplotlib
#use the below line for running the code on MAC OS
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from imageio import imread

import pdb  
"""
Utilities for making visualizations.
"""


def draw_layout(vocab, objs, boxes, masks=None, size=256,
                show_boxes=False, bgcolor=(0, 0, 0)):
  if bgcolor == 'white':
    bgcolor = (255, 255, 255)

  cmap = plt.get_cmap('rainbow')
  colors = cmap(np.linspace(0, 1, len(objs)))

  with torch.no_grad():
    # objs = objs.cpu().clone()
    # boxes = boxes.cpu().clone()
    boxes *= size
    
    if masks is not None:
      masks = masks.cpu().clone()
    
    bgcolor = np.asarray(bgcolor)
    bg = np.ones((size, size, 1)) * bgcolor
    plt.imshow(bg.astype(np.uint8))

    plt.gca().set_xlim(0, size)
    plt.gca().set_ylim(size, 0)
    plt.gca().set_aspect(1.0, adjustable='box')
    
    import pdb

    for i in range(len(objs)):
      name = objs[i]  
      name = vocab['object_idx_to_name'][objs[i]]  ## subarna

      if name == '__image__':
        continue
      box = boxes[i]

      if masks is None:
        continue
      mask = masks[i].numpy()
      mask /= mask.max()

      r, g, b, a = colors[i]
      colored_mask = mask[:, :, None] * np.asarray(colors[i])
      
      x0, y0, x1, y1 = box
      plt.imshow(colored_mask, extent=(x0, x1, y1, y0),
                 interpolation='bicubic', alpha=1.0)

    if show_boxes:
      for i in range(len(objs)):
        name = objs[i] 
        name = vocab['object_idx_to_name'][objs[i]]  ## subarna
        if name == '__image__':
          continue
        box = boxes[i]

        draw_box(box, colors[i], name)


def draw_box(box, color, text=None):
  """
  Draw a bounding box using pyplot, optionally with a text box label.

  Inputs:
  - box: Tensor or list with 4 elements: [x0, y0, x1, y1] in [0, W] x [0, H]
         coordinate system.
  - color: pyplot color to use for the box.
  - text: (Optional) String; if provided then draw a label for this box.
  """
  TEXT_BOX_HEIGHT = 10
  if torch.is_tensor(box) and box.dim() == 2:
    box = box.view(-1)
    assert box.size(0) == 4
  x0, y0, x1, y1 = box
  assert y1 > y0, box
  assert x1 > x0, box
  w, h = x1 - x0, y1 - y0
  rect = Rectangle((x0, y0), w, h, fc='none', lw=2, ec=color)
  plt.gca().add_patch(rect)
  if text is not None:
    text_rect = Rectangle((x0, y0), w, TEXT_BOX_HEIGHT, fc=color, alpha=0.5)
    plt.gca().add_patch(text_rect)
    tx = 0.5 * (x0 + x1)
    ty = y0 + TEXT_BOX_HEIGHT / 2.0
    plt.text(tx, ty, text, va='center', ha='center')


def draw_scene_graph(objs, triples, vocab=None, **kwargs):
  """
  Use GraphViz to draw a scene graph. If vocab is not passed then we assume
  that objs and triples are python lists containing strings for object and
  relationship names.
  Using this requires that GraphViz is installed. On Ubuntu 16.04 this is easy:
  sudo apt-get install graphviz
  """
  output_filename = kwargs.pop('output_filename', 'graph.png')
  orientation = kwargs.pop('orientation', 'V')
  edge_width = kwargs.pop('edge_width', 6)
  arrow_size = kwargs.pop('arrow_size', 1.5)
  binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
  ignore_dummies = kwargs.pop('ignore_dummies', True)
  
  if orientation not in ['V', 'H']:
    raise ValueError('Invalid orientation "%s"' % orientation)
  rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

  if vocab is not None:
    # Decode object and relationship names
    assert torch.is_tensor(objs)
    assert torch.is_tensor(triples)
    objs_list, triples_list = [], []
    for i in range(objs.size(0)):
      objs_list.append(vocab['object_idx_to_name'][objs[i].item()])
    for i in range(triples.size(0)):
      s = triples[i, 0].item()
      p = vocab['pred_name_to_idx'][triples[i, 1].item()]
      o = triples[i, 2].item()
      print("******", vocab)
      triples_list.append([s, p, o])
    objs, triples = objs_list, triples_list

  # General setup, and style for object nodes
  lines = [
    'digraph{',
    'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
    'rankdir=%s' % rankdir,
    'nodesep="0.5"',
    'ranksep="0.5"',
    'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
    'node [fillcolor="lightpink1"]',
  ]
  # Output nodes for objects
  for i, obj in enumerate(objs):
    if ignore_dummies and obj == '__image__':
      continue
    lines.append('%d [label="%s"]' % (i, obj))

  # Output relationships
  next_node_id = len(objs)
  lines.append('node [fillcolor="lightblue1"]')
  for s, p, o in triples:
    if ignore_dummies and p == '__in_image__':
      continue
    lines += [
      '%d [label="%s"]' % (next_node_id, p),
      '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
        s, next_node_id, edge_width, arrow_size, binary_edge_weight),
      '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
        next_node_id, o, edge_width, arrow_size, binary_edge_weight)
    ]
    next_node_id += 1
  lines.append('}')

  # Now it gets slightly hacky. Write the graphviz spec to a temporary
  # text file
  ff, dot_filename = tempfile.mkstemp()
  with open(dot_filename, 'w') as f:
    for line in lines:
      f.write('%s\n' % line)
  os.close(ff)

  # Shell out to invoke graphviz; this will save the resulting image to disk,
  # so we read it, delete it, then return it.
  output_format = os.path.splitext(output_filename)[1][1:]
  os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
  os.remove(dot_filename)
  img = imread(output_filename)
  os.remove(output_filename)

  return img  


##############################
def draw_scene_graph_temp_1(objs, triples, tot_obj, vocab=None, **kwargs):
  """
  Use GraphViz to draw a scene graph. If vocab is not passed then we assume
  that objs and triples are python lists containing strings for object and
  relationship names.

  Using this requires that GraphViz is installed. On Ubuntu 16.04 this is easy:
  sudo apt-get install graphviz
  """
  output_filename = kwargs.pop('output_filename', 'graph.png')
  orientation = kwargs.pop('orientation', 'V')
  edge_width = kwargs.pop('edge_width', 6)
  arrow_size = kwargs.pop('arrow_size', 1.5)
  binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
  ignore_dummies = kwargs.pop('ignore_dummies', True)
  
  if orientation not in ['V', 'H']:
    raise ValueError('Invalid orientation "%s"' % orientation)
  rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

  if vocab is not None:
    # Decode object and relationship names
    assert torch.is_tensor(objs)
    assert torch.is_tensor(triples)
    objs_list, triples_list = [], []
    for i in range(objs.size(0)):
      objs_list.append(vocab['object_idx_to_name'][objs[i].item()])

    for i in range(triples.size(0)):
      s = triples[i, 0].item()
      # p = vocab['pred_name_to_idx'][triples[i, 1].item()]
      p = vocab['pred_idx_to_name'][triples[i, 1].item()]
      ##########################
      # p = 'none'
      # for key, val in vocab['pred_name_to_idx'].items():
      #   if val == triples[i, 1].item():
      #     p = key
      ##########################    
      o = triples[i, 2].item()
      # triples_list.append([s, p, o])
      triples_list.append([s-tot_obj, p, o-tot_obj])
    objs, triples = objs_list, triples_list

  # General setup, and style for object nodes
  lines = [
    'digraph{',
    'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
    'rankdir=%s' % rankdir,
    'nodesep="0.5"',
    'ranksep="0.5"',
    'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
    'node [fillcolor="lightpink1"]',
  ]
  # Output nodes for objects
  for i, obj in enumerate(objs):
    if ignore_dummies and obj == '__image__':
      continue
    lines.append('%d [label="%s"]' % (i, obj))

  # Output relationships
  next_node_id = len(objs)
  lines.append('node [fillcolor="lightblue1"]')
  for s, p, o in triples:
    if ignore_dummies and p == '__in_image__':
      continue
    lines += [
      '%d [label="%s"]' % (next_node_id, p),
      '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
        s, next_node_id, edge_width, arrow_size, binary_edge_weight),
      '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
        next_node_id, o, edge_width, arrow_size, binary_edge_weight)
    ]
    next_node_id += 1
  lines.append('}')

  # Now it gets slightly hacky. Write the graphviz spec to a temporary
  # text file
  ff, dot_filename = tempfile.mkstemp()
  with open(dot_filename, 'w') as f:
    for line in lines:
      f.write('%s\n' % line)
  os.close(ff)

  # Shell out to invoke graphviz; this will save the resulting image to disk,
  # so we read it, delete it, then return it.
  output_format = os.path.splitext(output_filename)[1][1:]
  os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
  os.remove(dot_filename)
  img = imread(output_filename)
  os.remove(output_filename)

  return img


##############################
def draw_scene_graph_temp(objs, triples, tot_obj, vocab=None, **kwargs):
  """
  Use GraphViz to draw a scene graph. If vocab is not passed then we assume
  that objs and triples are python lists containing strings for object and
  relationship names.
  Using this requires that GraphViz is installed. On Ubuntu 16.04 this is easy:
  sudo apt-get install graphviz
  """
  geometric = ['next to', 'above', 'beside', 'behind', 'by', 'laying on', 'hanging on', 'under','on side of', 'below', 'against', 'attached to', 'parked on', 'on top of', 'at', 'on', 'in front of', 'near', 'along', 'around'  ]
  possessive = ['has', 'belonging to', 'have', 'with', 'covered in', 'inside', 'in', 'over', 'wears', 'wearing' ]
  misc = ['and', 'for', 'of', 'made of']
  semantic = ['covering', 'eating', 'standing on', 'holding', 'carrying', 'looking at', 'walking on', 'sitting on', 'riding', 'sitting in', 'standing in']

  output_filename = kwargs.pop('output_filename', 'graph.png')
  orientation = kwargs.pop('orientation', 'V')
  edge_width = kwargs.pop('edge_width', 6)
  arrow_size = kwargs.pop('arrow_size', 1.5)
  binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
  ignore_dummies = kwargs.pop('ignore_dummies', True)
  
  if orientation not in ['V', 'H']:
    raise ValueError('Invalid orientation "%s"' % orientation)
  rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

  if vocab is not None:
    # Decode object and relationship names
    assert torch.is_tensor(objs)
    assert torch.is_tensor(triples)
    objs_list, triples_list = [], []
    for i in range(objs.size(0)):
      objs_list.append(vocab['object_idx_to_name'][objs[i].item()])

    for i in range(triples.size(0)):
      s = triples[i, 0].item()
      p = vocab['pred_idx_to_name'][triples[i, 1].item()]
      ##########################
      # p = 'none'
      # for key, val in vocab['pred_name_to_idx'].items():
      #   if val == triples[i, 1].item():
      #     p = key
      ##########################    
      o = triples[i, 2].item()
      # triples_list.append([s, p, o])
      triples_list.append([s-tot_obj, p, o-tot_obj])
    objs, triples = objs_list, triples_list

  # General setup, and style for object nodes
  lines = [
    'digraph{',
    'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
    'rankdir=%s' % rankdir,
    'nodesep="0.5"',
    'ranksep="0.5"',
    'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
    'node [fillcolor="lightpink1"]',
  ]
  # Output nodes for objects
  for i, obj in enumerate(objs):
    if ignore_dummies and obj == '__image__':
      continue
    lines.append('%d [label="%s"]' % (i, obj))
  # Output relationships
  next_node_id = len(objs)
 # lines.append('node [fillcolor="lightblue1"]')
  for s, p, o in triples:
    if p in geometric:
        edge_color="green"
    elif p in possessive:
        edge_color="orange"
    elif p in semantic:
        edge_color="lightskyblue"
    elif p in misc:
        edge_color="grey"
    else:
        edge_color="grey"
    if ignore_dummies and p == '__in_image__':
      continue
    lines.append('node [fillcolor="%s"]' % (edge_color))
    lines += [
      '%d [label="%s"]' % (next_node_id, p),
      '%d->%d [penwidth=%f,arrowsize=%f,weight=%f,color=%s]' % (
        s, next_node_id, edge_width, arrow_size, binary_edge_weight,edge_color),
      '%d->%d [penwidth=%f,arrowsize=%f,weight=%f,color=%s]' % (
        next_node_id, o, edge_width, arrow_size, binary_edge_weight,edge_color)
    ]
    next_node_id += 1
  lines.append('}')

  # Now it gets slightly hacky. Write the graphviz spec to a temporary
  # text file
  ff, dot_filename = tempfile.mkstemp()
  with open(dot_filename, 'w') as f:
    for line in lines:
      f.write('%s\n' % line)
  os.close(ff)  

   # Shell out to invoke graphviz; this will save the resulting image to disk,
  # so we read it, delete it, then return it.
  output_format = os.path.splitext(output_filename)[1][1:]
  os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
  os.remove(dot_filename)
  img = imread(output_filename)
  os.remove(output_filename)

  return img



def draw_scene_graph_json(im_id, objs, triples, tot_obj, vocab=None, **kwargs):
  """
  Use GraphViz to draw a scene graph. If vocab is not passed then we assume
  that objs and triples are python lists containing strings for object and
  relationship names.
  Using this requires that GraphViz is installed. On Ubuntu 16.04 this is easy:
  sudo apt-get install graphviz
  """
  geometric = ['next to', 'above', 'beside', 'behind', 'by', 'laying on', 'hanging on', 'under','on side of', 'below', 'against', 'attached to', 'parked on', 'on top of', 'at', 'on', 'in front of', 'near', 'along', 'around'  ]
  possessive = ['has', 'belonging to', 'have', 'with', 'covered in', 'inside', 'in', 'over', 'wears', 'wearing' ]
  misc = ['and', 'for', 'of', 'made of']
  semantic = ['covering', 'eating', 'standing on', 'holding', 'carrying', 'looking at', 'walking on', 'sitting on', 'riding', 'sitting in', 'standing in']

  output_filename = kwargs.pop('output_filename', 'graph.png')
  orientation = kwargs.pop('orientation', 'V')
  edge_width = kwargs.pop('edge_width', 6)
  arrow_size = kwargs.pop('arrow_size', 1.5)
  binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
  ignore_dummies = kwargs.pop('ignore_dummies', True)
  
  if orientation not in ['V', 'H']:
    raise ValueError('Invalid orientation "%s"' % orientation)
  rankdir = {'H': 'LR', 'V': 'TD'}[orientation]


  import json
  data = {}

  if vocab is not None:
    # Decode object and relationship names
    assert torch.is_tensor(objs)
    assert torch.is_tensor(triples)
    objs_list, triples_list = [], []
    for i in range(objs.size(0)):
      objs_list.append(vocab['object_idx_to_name'][objs[i].item()])

    for i in range(triples.size(0)):
      s = triples[i, 0].item()
      p = vocab['pred_idx_to_name'][triples[i, 1].item()]
      ##########################
      o = triples[i, 2].item()
      # triples_list.append([s, p, o])
      triples_list.append([s-tot_obj, p, o-tot_obj])
    objs, triples = objs_list, triples_list

  # General setup, and style for object nodes
  lines = [
    'digraph{',
    'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
    'rankdir=%s' % rankdir,
    'nodesep="0.5"',
    'ranksep="0.5"',
    'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
    'node [fillcolor="lightpink1"]',
  ]
  # Output nodes for objects
  for i, obj in enumerate(objs):
    if ignore_dummies and obj == '__image__':
      continue
    lines.append('%d [label="%s"]' % (i, obj))
  # Output relationships
  next_node_id = len(objs)
 # lines.append('node [fillcolor="lightblue1"]')

  r = -1
  data['im_id'] = str('%06d' % im_id)

  for s, p, o in triples:
    if p in geometric:
        edge_color="green"
        data['relation'] = 2
    elif p in possessive:
        edge_color="orange"
        data['relation'] = 3
    elif p in semantic:
        edge_color="lightskyblue"
        data['relation'] = 1
    elif p in misc:
        edge_color="grey"
        data['relation'] = 4
    else:
        edge_color="grey"
        data['relation'] = 4

    r = r+1    
    if ignore_dummies and p == '__in_image__':
      continue
    
    data['caption'] = vocab['object_idx_to_name'][s] + ' ' + p + ' ' + vocab['object_idx_to_name'][o]
    data['idx'] = r
    
    if vocab['object_idx_to_name'][s] != "__image__" and vocab['object_idx_to_name'][o] != "__image__":
      ####### write the json file ###############
      with open('vg_captions_500.json', 'a') as f:
        json.dump(data, f)
        f.write(',')
    ###########################################
    

    lines.append('node [fillcolor="%s"]' % (edge_color))
    lines += [
      '%d [label="%s"]' % (next_node_id, p),
      '%d->%d [penwidth=%f,arrowsize=%f,weight=%f,color=%s]' % (
        s, next_node_id, edge_width, arrow_size, binary_edge_weight,edge_color),
      '%d->%d [penwidth=%f,arrowsize=%f,weight=%f,color=%s]' % (
        next_node_id, o, edge_width, arrow_size, binary_edge_weight,edge_color)
    ]
    next_node_id += 1
  lines.append('}')

  # Now it gets slightly hacky. Write the graphviz spec to a temporary
  # text file
  ff, dot_filename = tempfile.mkstemp()
  with open(dot_filename, 'w') as f:
    for line in lines:
      f.write('%s\n' % line)
  os.close(ff)  

   # Shell out to invoke graphviz; this will save the resulting image to disk,
  # so we read it, delete it, then return it.
  output_format = os.path.splitext(output_filename)[1][1:]
  os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
  os.remove(dot_filename)
  img = imread(output_filename)
  os.remove(output_filename)



def drawrect(drawcontext, xy, outline=None, width=1):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)



def overlay_boxes(imgs, vocab, objs, layout_boxes_t, obj_to_img, W=64, H=64, drawText=True, drawSuperbox=False):
  from PIL import Image, ImageDraw, ImageFont
  import pdb       
  from torchvision.utils import save_image
  draw_mask = True

  overlaid_boxes = []

  alpha = 0.5
  #num_images = np.unique(obj_to_img.numpy())
  num_images = np.unique(obj_to_img.cpu().numpy())
  colors = ['red', 'yellow', 'green', 'orange', 'blue', 'cyan', 'magenta', 'violet' ]
  n_cols = len(colors)
  # font = ImageFont.truetype("sans-serif.ttf", 16)
  no_show_obj_list = ['playingfield','fence', 'tree','pavement','floor_other','wall_other',
                      'sky-other', 'grass', 'wall-brick', 'branch', 'snow', 'platform', 
                      'clouds', 'building-other', 'wall-other' 
                      ]


  # Z = (imgs[0].numpy().transpose(1, 2, 0) * 255.0)
  #Z1 = imgs[0] * 255.0
  Z1 = imgs * 255.0
 
  if torch.is_tensor(Z1):
    Z = Z1.numpy()
  else:
    Z = Z1  

  img = Image.fromarray(Z.astype('uint8')).convert('RGBA')
  canvas = img.copy()

  if torch.is_tensor(layout_boxes_t):
    #layout_boxes = layout_boxes_t.numpy().copy()
    layout_boxes = layout_boxes_t.cpu().numpy().copy()
  else:  
    layout_boxes = layout_boxes_t.copy()

  # expand dimension to [:,4]
  layout_boxes = np.expand_dims(layout_boxes, axis=0)
  layout_boxes[:,0] *= W
  layout_boxes[:,2] *= W
  layout_boxes[:,1] *= H
  layout_boxes[:,3] *= H
  
  # boxes = layout_boxes.numpy()
  # boxes = np.trunc(boxes)
  boxes = np.trunc(layout_boxes)

  last_img = 0

  draw = ImageDraw.Draw(img)
  p = 0
  img_idx = 0

  for j in range(layout_boxes.shape[0]):      
    if obj_to_img[j] != last_img:
      overlaid_boxes.append(np.asarray(img))
      last_img = obj_to_img[j]

      img_idx += 1
      # Z = (imgs[img_idx].numpy().transpose(1, 2, 0) ) * 255.0)
      Z = imgs[img_idx] * 255.0
      
      if torch.is_tensor(Z):
          Z = Z.numpy()

      img = Image.fromarray(Z.astype('uint8')).convert('RGBA')
      draw = ImageDraw.Draw(img) 
      canvas = img.copy()
      p = 0  

    if(drawText):
      obj_name = None
      obj_name = vocab['object_idx_to_name'][objs[j]] # draw text
      if obj_name == '__image__':
        continue 
      if obj_name in no_show_obj_list:
        continue

    draw = ImageDraw.Draw(img)  
    # draw.rectangle(boxes[j], outline = colors[p%n_cols]) 
    # draw.rectangle(boxes[j], outline = colors[p%n_cols]) 

    if(drawSuperbox):
      c_outline = 'black' 
      drawrect(draw, [(boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3])], outline=c_outline, width=1)
    else:        
      c_outline = colors[p%n_cols]
      drawrect(draw, [(boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3])], outline=c_outline, width=1)
    
    if(drawText):
      draw.text((boxes[j,0]+8, boxes[j,1]+8), obj_name, c_outline)  # draw text
    
    del draw
    p += 1

  overlaid_boxes.append(np.asarray(img))
  return overlaid_boxes
############################## 

def debug_layout_mask(vocab, objs, layout_boxes_t, layout_masks, obj_to_img, W=256, H=256):
  from PIL import Image, ImageDraw, ImageFont     
  from torchvision.utils import save_image
  draw_mask = True

  layouts = []

  alpha = 0.5
  num_images = np.unique(obj_to_img.numpy())
  colors = ['red', 'yellow', 'green', 'orange', 'blue', 'cyan', 'magenta', 'violet' ]
  n_cols = len(colors)
  # font = ImageFont.truetype("sans-serif.ttf", 16)
  no_show_obj_list = ['playingfield','fence', 'tree','pavement','floor_other','wall_other',
                      'sky-other', 'grass', 'wall-brick', 'branch', 'snow', 'platform', 
                      'clouds', 'building-other', 'wall-other' 
                      ]

  Z = np.ones((W,H,3)) #*255 
  img = Image.fromarray(Z.astype('uint8')).convert('RGBA')
  canvas = img.copy()
  # img = Image.new("RGB", (W, H), "white")

  if torch.is_tensor(layout_boxes_t):
    layout_boxes = layout_boxes_t.numpy().copy()
  else:
    layout_boxes = layout_boxes_t.copy()
  
  layout_boxes[:,0] *= W
  layout_boxes[:,2] *= W
  layout_boxes[:,1] *= H
  layout_boxes[:,3] *= H
  # boxes = layout_boxes.numpy()
  # boxes = np.trunc(boxes)
  boxes = np.trunc(layout_boxes)

  layout_masks = layout_masks.numpy()
  layout_masks = layout_masks / layout_masks.max()
  layout_masks *= 255

  last_img = 0

  draw = ImageDraw.Draw(img)
  p = 0
  for j in range(layout_boxes.shape[0]):      
    if obj_to_img[j] != last_img:
      layouts.append(np.asarray(img))
      last_img = obj_to_img[j]
      img = Image.fromarray(Z.astype('uint8')).convert('RGBA')
      draw = ImageDraw.Draw(img) 
      canvas = img.copy()
      p = 0  

    obj_name = vocab['object_idx_to_name'][objs[j]]
    if obj_name == '__image__':
      continue
    
    if obj_name in no_show_obj_list:
      continue
    #######################
    if draw_mask:
      layout_mask_c = layout_masks[j]
      layout_mask_c = 255 - layout_mask_c
      mask_s = Image.fromarray(layout_mask_c.astype('uint8')).convert('RGBA')
      
      ################################
      ### colored mask ###
      mask_sk = Image.new("RGB", (layout_mask_c.shape), colors[p%n_cols])
      pixdata = mask_sk.load()
      width, height = mask_sk.size
      for y in range(height):
          for x in range(width):
            r,g,b = pixdata[x,y]
            #pixdata[x, y] = (r-np.int(layout_mask_c[x, y]), g-np.int(layout_mask_c[x, y]), b-np.int(layout_mask_c[x, y]))
            pixdata[x, y] = (r-np.int(layout_mask_c[y, x]), g-np.int(layout_mask_c[y, x]), b-np.int(layout_mask_c[y, x]))
      ################################

      mask_w = boxes[j,2] - boxes[j,0]
      mask_h = boxes[j,3] - boxes[j,1]
      if mask_w < W/20 or mask_h < H/20:
        continue
      elif mask_w >= W and mask_h >= H:
        continue 
      else:
        mask = mask_sk.resize([mask_w, mask_h])
        canvas.paste(mask, [np.int64(boxes[j,0]), np.int64(boxes[j,1])])

        c_arr = np.array(canvas)
        i_arr = np.array(img)
        c_arr = np.maximum(c_arr, i_arr)
        img = Image.fromarray(c_arr.astype('uint8')).convert('RGBA')

        draw = ImageDraw.Draw(img)

  
    draw.rectangle(boxes[j], outline = colors[p%n_cols]) 
    draw.text((boxes[j,0]+8, boxes[j,1]+8), obj_name, colors[p%n_cols]) 
    p += 1

  layouts.append(np.asarray(img))
  return layouts
##############################     


if __name__ == '__main__':
  o_idx_to_name = ['cat', 'dog', 'hat', 'skateboard']
  p_idx_to_name = ['riding', 'wearing', 'on', 'next to', 'above']
  o_name_to_idx = {s: i for i, s in enumerate(o_idx_to_name)}
  p_name_to_idx = {s: i for i, s in enumerate(p_idx_to_name)}
  vocab = {
    'object_idx_to_name': o_idx_to_name,
    'object_name_to_idx': o_name_to_idx,
    'pred_idx_to_name': p_idx_to_name,
    'pred_name_to_idx': p_name_to_idx,
  }

  objs = [
    'cat',
    'cat',
    'skateboard',
    'hat',
  ]
  objs = torch.LongTensor([o_name_to_idx[o] for o in objs])
  triples = [
    [0, 'next to', 1],
    [0, 'riding', 2],
    [1, 'wearing', 3],
    [3, 'above', 2],
  ]
  triples = [[s, p_name_to_idx[p], o] for s, p, o in triples]
  triples = torch.LongTensor(triples)

  draw_scene_graph(objs, triples, vocab, orientation='V')

