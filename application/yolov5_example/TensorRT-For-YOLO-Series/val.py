from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from utils.utils import preproc, vis
from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os
import argparse

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 80  # your model classes

def validate_on_coco(engine_path, coco_annotation_path, coco_image_dir, conf=0.05, end2end=False):
    # 初始化 COCO API
    coco = COCO(coco_annotation_path)
    img_ids = coco.getImgIds()
    coco_results = []

    # 初始化预测器
    pred = Predictor(engine_path=engine_path)

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(coco_image_dir, img_info['file_name'])
        origin_img, dets = pred.inference(img_path, end2end=args.end2end)
        # 处理推理结果
        if len(dets[0]) > 0:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            for box, score, cls_id in zip(final_boxes, final_scores, final_cls_inds):
                x1, y1, x2, y2 = box
                coco_results.append({
                    "image_id": img_id,
                    "category_id": int(coco.getCatIds()[int(cls_id)]),  # COCO 类别从 1 开始
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score)
                })
        


        # # 可视化真实标签
        # annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        # gt_img = cv2.imread(img_path)
        # for ann in annotations:
        #     x, y, w, h = ann['bbox']
        #     category_id = ann['category_id']
        #     class_name = coco.loadCats(category_id)[0]['name']
        #     cv2.rectangle(gt_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        #     cv2.putText(gt_img, class_name, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # cv2.imwrite("test1.jpg", origin_img)
        # cv2.imwrite("test2.jpg", gt_img)

    # 保存结果为 JSON 文件
    result_json_path = "coco_results.json"
    with open(result_json_path, "w") as f:
        json.dump(coco_results, f)

    # 使用 COCO API 评估结果
    coco_dt = coco.loadRes(result_json_path)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default='/home/ynz/qat/yolov5/yolov5s_qat.engine', help="TRT engine Path")
    parser.add_argument("--coco_annotation", default='/home/ynz/datasets/coco/annotations/instances_val2017.json', help="Path to COCO annotation file")
    parser.add_argument("--coco_image_dir", default='/home/ynz/datasets/coco/images/val2017', help="Path to COCO validation images directory")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--end2end", default=False, action="store_true", help="Use end2end engine")
    args = parser.parse_args()

    validate_on_coco(
        engine_path=args.engine,
        coco_annotation_path=args.coco_annotation,
        coco_image_dir=args.coco_image_dir,
        conf=args.conf,
        end2end=args.end2end
    )