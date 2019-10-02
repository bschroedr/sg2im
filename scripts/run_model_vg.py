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

import argparse, json, os

from imageio import imwrite
import torch

from sg2im.model import Sg2ImModel
from sg2im.data.utils import imagenet_deprocess_batch
import sg2im.vis as vis


parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint', default='sg2im-models/vg128.pt')
# parser.add_argument('--checkpoint', default='sg2im-models/coco64.pt')
# parser.add_argument('--checkpoint', default='copied_from_server/SG_PF_MA_NP_stage2/checkpoint_with_model.pt')
parser.add_argument('--checkpoint', default='copied_from_server/SG_PF_MA_NP/checkpoint_with_model.pt')
parser.add_argument('--output_dir', default='outputs')


parser.add_argument('--scene_graphs_json', default='scene_graphs_test/coco_test.json')
# parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--draw_scene_graphs', type=int, default=1)
parser.add_argument('--overlay_boxes', type=int, default=1)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])


def main(args):
  if not os.path.isfile(args.checkpoint):
    print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
    print('Maybe you forgot to download pretraind models? Try running:')
    print('bash scripts/download_models.sh')
    return

  if not os.path.isdir(args.output_dir):
    print('Output directory "%s" does not exist; creating it' % args.output_dir)
    os.makedirs(args.output_dir)

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

  # Load the scene graphs
  with open(args.scene_graphs_json, 'r') as f:
    scene_graphs = json.load(f)

  # Run the model forward
  with torch.no_grad():
    # imgs, boxes_pred, masks_pred, _ = model.forward_json(scene_graphs)    
    imgs, boxes_pred, masks_pred, objs, layout, layout_boxes_t, layout_masks, obj_to_img, sg_context_pred, _, _ = model.forward_json(scene_graphs)
  imgs = imagenet_deprocess_batch(imgs)

  layout_boxes = layout_boxes_t.numpy()

  np_imgs = []
  # Save the generated images
  import numpy as np
  for i in range(imgs.shape[0]):
    # img_np = imgs[i].numpy().transpose(1, 2, 0)
    img_np = (imgs[i].numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    img_path = os.path.join(args.output_dir, 'img%06d.png' % i)
    imwrite(img_path, img_np)
    np_imgs.append(img_np)

  # Draw the scene graphs
  if args.draw_scene_graphs == 1:
    for i, sg in enumerate(scene_graphs):
      sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
      sg_img_path = os.path.join(args.output_dir, 'sg%06d.png' % i)
      imwrite(sg_img_path, sg_img)


  # # # # draw the layout
  # layouts = vis.debug_layout_mask(model.vocab, objs, layout_boxes, layout_masks, obj_to_img, W=256, H=256)
  # for i, layout in enumerate(layouts):
  #   layout_path = os.path.join(args.output_dir, 'layout%06d.png' % i)  
  #   imwrite(layout_path, layout)    

  # Draw the boxes overlaid on image
  # if args.overlay_boxes == 1:
  #   overlaid_images = vis.overlay_boxes(imgs, model.vocab, objs, layout_boxes, obj_to_img, W=64, H=64)   
  #   for i, overlaid in enumerate(overlaid_images):
  #     overlaid_path = os.path.join(args.output_dir, 'overlaid%06d.png' % i)  
  #     imwrite(overlaid_path, overlaid)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

