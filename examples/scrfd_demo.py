from typing import Tuple, List, Dict
import os
import os.path as osp
import sys
import argparse
import cv2
from faceimagekit.face_detectors import scrfd_onnx
from faceimagekit.utils import draw_face, Timer, resize_image, rersize_points

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test scrfd face detection')
    # Basic
    parser.add_argument('-weight', '--weight_path', type=str, help="onnx weight file.")
    parser.add_argument('-hd', '--accelerator', type=str,
                        choices=['cpu', 'gpu'], default='cpu', help="hardware type.")
    parser.add_argument('-engine', '--engine_type', type=str,
                        choices=['ONNXInfer', 'NCNNInfer'], default='ONNXInfer', help="engine type.")
    parser.add_argument('--input_shape',type=list, default=[640, 640], help='resize input shape: h, w')
    parser.add_argument('--threshold',type=float, default=0.5, help='score threshold')
    parser.add_argument('--nms',type=float, default=0.4, help='nms threshold')
    parser.add_argument('-files', '--file_list',
                        type=list, nargs='+', default=[0], help="file path list")
    parser.add_argument('--save_path',type=str, help='path to save generation result')
    parser.add_argument('--imshow', action='store_true', help="show image with opencv")
    return parser.parse_args()

def main():
    args = parse_args()
    if osp.exists(args.weight_path):
        raise FileNotFoundError(f"can't found {args.weight_path}")
    infer = scrfd_onnx(args.weight_path, backend=args.engine_type)
    infer.prepare(nms_threshold=args.nms)
    
    for fp in args.file_list:
        t_im = Timer()
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        res_img = resize_image(img, args.input_shape[::-1])
        print(f"read img time: {t_im.time()} s")
        dets_list, kpss_list = infer.detect(img, threshold=args.threshold)
        results = []
        for dets, kps in zip(dets_list, kpss_list):
            results.append(
                {
                    'bbox': dets[0: 4],
                    'prob': dets[4],
                    'landmarks': kps
                }
            )
        show_img = draw_face(img, results, draw_socre=True)
        if args.imshow:
            cv2.imshow(f"{osp.basename(fp)}", show_img)
            cv2.waitKey(0)
        if args.save_path:
            if not osp.exists(args.save_path):
                os.makedirs(args.save_path)
            cv2.imwrite(osp.join(args.save_path, osp.basename(fp)), show_img)
        

    
if __name__ == "__main__":
    main()