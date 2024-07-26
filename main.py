import os
import shutil

import argparse

import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS

import numpy as np
import torch
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmengine.visualization.utils import tensor2ndarray

import inpaint
import getframe

def get_video_fps(video_path):
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
      print(f"Error: Could not open video {video_path}")
      return None
   fps = cap.get(cv2.CAP_PROP_FPS)
   cap.release()
   return fps

def get_codec(cap):
   fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
   codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
   return codec

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    #parser.add_argument('framedir', help='Frame directory')
    parser.add_argument('video', help='Video path')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.7, help='Bbox score threshold')
    args = parser.parse_args()
    return args

def main():
    """
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # Convert video to frame images
    framedir = getframe.getframe(args.video)

    maskframedir = f"mask-{framedir}"
    os.makedirs(f"{maskframedir}", exist_ok=True)

    filenum = len([name for name in os.listdir(framedir) if os.path.isfile(os.path.join(framedir, name))])
    for no in range(filenum):
        nostr = str(no).zfill(5)
        path = f"{framedir}/{nostr}.png"
        print(f"[2/3]Executing instance segmentation {no}/{filenum-1}")
        frame = cv2.imread(path)
        # Execute instance segmentation
        result = inference_detector(model, frame, test_pipeline=test_pipeline)

        # Get human mask image from instance sermentation result
        result = result.cpu()
        pred_instances = result.pred_instances
        pred_instances = pred_instances[pred_instances.scores > args.score_thr]
        pred_instances = pred_instances[pred_instances.labels == 0]
        masks = pred_instances.masks
        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()
        elif isinstance(masks, (PolygonMasks, BitmapMasks)):
            masks = masks.to_ndarray()
        
        masks = masks.astype(bool)
        masks = tensor2ndarray(masks)
        masks = masks.astype('uint8') * 255

        res = np.zeros_like(frame)

        for mask in masks:
            rgb = np.zeros_like(frame)
            rgb[...] = (1, 1, 1)
            rgb = cv2.bitwise_and(rgb, rgb, mask=mask)
            res = res + rgb
        
        _, res = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY)
        res = res[:, :, 0]

        cv2.imwrite(f'{maskframedir}/{nostr}.png', np.array(res, dtype='uint8'))

    fps = get_video_fps(args.video)
    outfile = f"mask-{framedir}.mp4"
    """
    framedir = "20240525_114131000_iOS"
    maskframedir = "mask-20240525_114131000_iOS"
    fps = 60
    outfile = f"mask-{framedir}.mp4"
    inpaint.inpaint(framedir, maskframedir, outfile, fps)
    # Post-processing
    print("Post-processing")
    # Remove directory
    # shutil.rmtree(framedir)
    # shutil.rmtree(maskframedir)

if __name__ == '__main__':
    main()
