from typing import Tuple, List, Dict
import os
import os.path as osp
import sys
import platform
import argparse
import pathlib
import cv2
from faceimagekit.face_segmenters import ppliteseg_model
from faceimagekit.utils import draw_face, Timer, resize_image, rersize_points

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test face segmentation detection')
    # Basic
    parser.add_argument('-weight', '--weight_path', type=str, help="onnx weight file.")
    parser.add_argument('-hd', '--accelerator', type=str,
                        choices=['cpu', 'gpu'], default='cpu', help="hardware type.")
    parser.add_argument('-engine', '--engine_type', type=str,
                        choices=['ONNXInfer', 'NCNNInfer'], default='ONNXInfer', help="engine type.")
    parser.add_argument('--input_shape',type=int, nargs='+', default=[3, 512, 512], help='resize input shape: c, h, w')
    parser.add_argument('--threshold',type=float, default=0.5, help='score threshold')
    parser.add_argument('-files', '--file_list',
                        type=str, nargs='+', default=[], help="file path list")
    parser.add_argument('--save_path',type=str, help='path to save generation result')
    parser.add_argument('--imshow', action='store_true', help="show image with opencv")
    return parser.parse_args()

def main():
    args = parse_args()

    infer = ppliteseg_model(args.weight_path, backend=args.engine_type, input_shape=args.input_shape)
    infer.prepare(device=args.accelerator)
    
    for fp in args.file_list:
        t_im = Timer()
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise FileExistsError(f"opencv read {str(fp)} failed")
        
        print(f"read img time: {t_im.time()} s")
        
        t_infer = Timer()
        seg_color_mask = infer.predict(img, palette=True)
        print(f"model name: {args.weight_path}, input_shape: {args.input_shape}, infer time: {t_infer.time()} s")

        if args.imshow:
            show_name = osp.basename(fp)
            if min(seg_color_mask.shape[0: 2]) > 1080:
                h, w = seg_color_mask.shape[0: 2]
                resize = (int(w*0.5), int(h*0.5))
                show_mask = cv2.resize(seg_color_mask, resize, interpolation=cv2.INTER_LINEAR)
                cv2.imshow(show_name, show_mask)
            else:
                cv2.imshow(show_name, seg_color_mask)
            cv2.waitKey(0)
        if args.save_path:
            if not osp.exists(args.save_path):
                os.makedirs(args.save_path)
            cv2.imwrite(osp.join(args.save_path, osp.basename(fp)), seg_color_mask)
        

    
if __name__ == "__main__":
    main()