from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

import sys
from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result

os.environ["PYOPENGL_PLATFORM"] = "egl"

DETECTRON_CFG = 'llib/utils/keypoints/configs/cascade_mask_rcnn_vitdet_h_75ep.py'
DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"

class ViTPoseModel(object):
    # MODEL_DICT = {
    #     'ViTPose+-G (multi-task train, COCO)': {
    #         'config': f'third-party/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py',
    #         'model': f'essentials/vitpose/vitpose-h.pth',
    #     },
    # }
    MODEL_DICT = {
        'ViTPose+-G (multi-task train, COCO)': {
            'config': f'third-party/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py',
            'model': f'essentials/vitpose/wholebody.pth',
        },
    }
    def __init__(
            self, 
            device: str | torch.device,
            model_name: str = 'ViTPose+-G (multi-task train, COCO)',
        ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic['model']
        model = init_pose_model(dic['config'], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        image = image[:, :, ::-1]  # RGB -> BGR
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  # BGR -> RGB

if __name__ == "__main__":
    import sys
    from pathlib import Path
    # sys.path.insert(0, '/content/buddi/third-party/4D-Humans-Keypoints')
    # from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from llib.utils.keypoints.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    # import hmr2
    import cv2
    import json
    import argparse 

    parser = argparse.ArgumentParser(description='Run ViTPose on images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run ViTPose on')
    parser.add_argument('--out_folder', type=str, default='demo/data/FlickrCI3D_Signatures/demo/vitpose_live', help='Output folder for keypoints')
    parser.add_argument('--image_folder', type=str, default='demo/data/FlickrCI3D_Signatures/demo/images', help='Input folder for images')
    parser.add_argument('--vitpose_model_name', type=str, default='ViTPose+-G (multi-task train, COCO)', help='ViTPose model type')
    parser.add_argument('--detectron_cfg', type=str, default=DETECTRON_CFG, help='Detectron2 config file')
    parser.add_argument('--detectron_url', type=str, default=DETECTRON_URL, help='Detectron2 model URL')
    args = parser.parse_args()

    device = args.device 
    vitpose_model_type = args.vitpose_model_name
    image_folder = args.image_folder
    out_folder = args.out_folder
    os.makedirs(out_folder, exist_ok=True)


    cfg_path = Path(args.detectron_cfg)
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = args.detectron_url
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25

    detector = DefaultPredictor_Lazy(detectron2_cfg)


    # ignore unexpected key in source state_dict
    cpm = ViTPoseModel(device, model_name=vitpose_model_type)

    # Make output directory if it does not exist
    os.makedirs(out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_fns = os.listdir(image_folder)

    # Iterate over all images in folder
    for img_fn in img_fns:
        img_path = os.path.join(image_folder, img_fn)
        img_cv2 = cv2.imread(img_path)

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        json_contents = {}
        json_contents['people'] = []

        # We bring ViTPose keypoints into OpenPose format
        for person in vitposes_out:
            keypoints = person['keypoints'].astype(np.float64)
            keypoints_body = keypoints[:17,:]
            keypoints_left_hand = keypoints[-42:-21,:].reshape(-1)
            keypoints_right_hand = keypoints[-21:].reshape(-1)
            keypoints_face = np.concatenate((keypoints[23:-42].reshape(-1), np.zeros(6)))
            body_pose = np.zeros([25,3])
            body_pose[[0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]] = keypoints_body
            body_pose[[19,20,21]] = keypoints[[17,18,19],:] # left foot
            body_pose[[22,23,24]] = keypoints[[20,21,22],:] # right foot
            body_pose = np.reshape(body_pose, -1)
            
            json_contents['people'].append({'pose_keypoints_2d':list(body_pose),
                                            'hand_left_keypoints_2d':list(keypoints_left_hand),
                                            'hand_right_keypoints_2d':list(keypoints_right_hand),
                                            'face_keypoints_2d': list(keypoints_face)})

        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        f = open(os.path.join(out_folder, f'{img_fn}_keypoints.json'), 'w')
        json.dump(json_contents, f)
        f.close()