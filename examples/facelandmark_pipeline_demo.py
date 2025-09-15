from typing import Tuple, List, Dict
import os
import os.path as osp
import sys
import argparse
import pathlib
import numpy as np
import cv2
from faceimagekit.pipelines import FaceLandmarkPipeline
from faceimagekit.utils import draw_face, Timer, resize_image, rersize_points


def parse_args():
    parser = argparse.ArgumentParser(description='Test FaceLandmarkPipeline')
    # Basic
    parser.add_argument('-det_weight', '--det_weight_path',
                        type=str, help="det weight file.")
    parser.add_argument('-ld_weight', '--ld_weight_path',
                        type=str, help="landmark weight file.")
    parser.add_argument('-hd', '--accelerator', type=str,
                        choices=['cpu', 'gpu'], default='cpu', help="hardware type.")
    parser.add_argument('-det_engine', '--det_engine_type', type=str,
                        choices=['ONNXInfer', 'NCNNInfer'], default='ONNXInfer', help="detection engine type.")
    parser.add_argument('-ld_engine', '--ld_engine_type', type=str,
                        choices=['OpencvInfer'], default='OpencvInfer', help="landmark engine type.")
    parser.add_argument('--det_input_shape', type=int, nargs='+',
                        default=[3, 640, 640], help='detector input shape: c, h, w')
    parser.add_argument('--ld_input_shape', type=int, nargs='+',
                        default=[256, 256, 3], help='landmarker input shape: h, w, c')
    parser.add_argument('--threshold', type=float,
                        default=0.5, help='score threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='nms threshold')
    parser.add_argument('-files', '--file_list',
                        type=str, nargs='+', default=[], help="file path list")
    parser.add_argument('--save_path', type=str,
                        help='path to save generation result')
    parser.add_argument('--imshow', action='store_true',
                        help="show image with opencv")
    return parser.parse_args()


def main():
    args = parse_args()
    infer = FaceLandmarkPipeline(
        args.det_weight_path,
        args.ld_weight_path,
        args.det_engine_type,
        args.ld_engine_type,
        args.det_input_shape,
        args.ld_input_shape,
        args.accelerator
    )
    try:
        infer.prepare()
    except Exception as e:
        raise RuntimeError(f"FaceLandmarkPipeline infer error: {str(e)}")

    for fp in args.file_list:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise FileExistsError(f"opencv read {str(fp)} failed")

        t_infer = Timer()
        face_list = infer.predict(
            img, score_threshold=args.threshold, nms_threshold=args.nms)
        print(f"FaceLandmarkPipeline infer time: {t_infer.time()} s")
        show_img = draw_face(
            img, face_list, draw_socre=True, draw_kps=True, draw_lanamrk=True)
        if args.imshow:
            show_name = osp.basename(fp)
            if min(show_img.shape[0: 2]) > 1080:
                h, w = show_img.shape[0: 2]
                resize = (int(w*0.5), int(h*0.5))
                resize_img = cv2.resize(
                    img, resize, interpolation=cv2.INTER_AREA)
                cv2.imshow(show_name, resize_img)
            else:
                cv2.imshow(show_name, show_img)
            cv2.waitKey(0)
        if args.save_path:
            if not osp.exists(args.save_path):
                os.makedirs(args.save_path)
            cv2.imwrite(osp.join(args.save_path, osp.basename(fp)), show_img)


if __name__ == "__main__":
    main()
