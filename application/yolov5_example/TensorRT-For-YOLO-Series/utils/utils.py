import tensorrt as trt
from cuda import cudart
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from utils import common 

import numpy as np
import torch
import torchvision.ops

def yolov5_postprocess(
    preds, 
    anchors, 
    strides=[8, 16, 32], 
    num_classes=80, 
    conf_thres=0.25, 
    iou_thres=0.6, 
    img_size=640,
    original_shape=None,
    ratio=1,
    dwdh=0,  # 原始图像尺寸（用于Letterbox还原）
):
    """
    Args:
        preds (list): 模型输出的三个特征图，每个形状为 (1, 3, grid_h, grid_w, 85)
        anchors (list): 锚框尺寸列表，例如 [[[10,13], [16,30], [33,23]], ...] 对应每个特征层
    """
    # 1. 合并三个特征图并解码坐标
    all_boxes = []
    all_scores = []
    all_labels = []
    for i in range(len(preds)):  # 遍历每个特征层
        feat = preds[i][0]  # 移除batch维度，形状为 (3, grid_h, grid_w, 85)
        grid_h, grid_w = feat.shape[1:3]
        anchor = np.array(anchors[i])  # 当前特征层的锚框尺寸
        stride = strides[i]
        
        # 创建网格坐标 (cx, cy)
        grid_x, grid_y = np.meshgrid(
            np.arange(grid_w),
            np.arange(grid_h)
        )
        grid_xy = np.stack([grid_x, grid_y], axis=-1).astype(np.float32) - 0.5 # (grid_h, grid_w, 2)
        
        # 解码坐标
        feat = sigmoid(feat)
        pred_boxes = feat[..., :4]  # (3, grid_h, grid_w, 4)
        pred_obj = feat[..., 4:5]   # 物体置信度
        pred_cls = feat[..., 5:]    # 类别概率
        
        # 中心坐标：bx = (sigmoid(tx) + grid_xy) * stride
        pred_xy = pred_boxes[...,:2] * 2 + grid_xy  # Sigmoid(tx, ty)
        pred_xy = pred_xy * stride
        
        # 宽高：bw = exp(tw) * anchor_w * stride，bh = exp(th) * anchor_h * stride
        pred_wh = (pred_boxes[...,2:] * 2) ** 2 * anchor[:, np.newaxis, np.newaxis, :]

        # 转换为xyxy格式
        pred_x1y1 = pred_xy - pred_wh * 0.5
        pred_x2y2 = pred_xy + pred_wh * 0.5
        pred_boxes = np.concatenate([pred_x1y1, pred_x2y2], axis=-1)
        
        # 扁平化处理（将3个锚框合并为一维）
        pred_boxes = pred_boxes.reshape(-1, 4)
        pred_obj = pred_obj.reshape(-1, 1)
        pred_cls = pred_cls.reshape(-1, num_classes)
        
        # 计算最终置信度：obj_conf * cls_conf
        scores = pred_cls
        
        max_scores = np.max(pred_obj, axis=-1)
        labels = np.argmax(scores, axis=-1)
        
        all_boxes.append(pred_boxes)
        all_scores.append(max_scores)
        all_labels.append(labels)
    
    # 合并所有特征层的结果
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # 2. 过滤低置信度框
    keep = scores > conf_thres
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # 3. 非极大值抑制（NMS）
    if boxes.shape[0] == 0:
        return [], [], []
    
    # 按类别分组
    unique_labels = np.unique(labels)
    final_boxes = []
    final_scores = []
    final_labels = []
    for label in unique_labels:
        idx = (labels == label)
        cls_boxes = boxes[idx]
        cls_scores = scores[idx]
        if cls_boxes.shape[0] == 0:
            continue
        
        # 按置信度排序
        sorted_idx = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[sorted_idx]
        cls_scores = cls_scores[sorted_idx]
        
        # NMS
        keep = nms(cls_boxes, cls_scores, iou_thres)
        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_labels.append(np.full_like(cls_scores[keep], label))
    
    if not final_boxes:
        return [], [], []
    
    # 合并最终结果
    boxes = np.concatenate(final_boxes, axis=0)
    scores = np.concatenate(final_scores, axis=0)
    labels = np.concatenate(final_labels, axis=0)
    boxes = scale_coords(boxes, ratio, dwdh, original_shape)
    # # 4. 还原坐标到原始图像尺寸（Letterbox处理）
    # if original_shape is not None:
    #     ratio = min(img_size / original_shape[0], img_size / original_shape[1])
    #     dw = (img_size - original_shape[1] * ratio) / 2
    #     dh = (img_size - original_shape[0] * ratio) / 2
    #     boxes = boxes * (1 / ratio)
    #     boxes[:, [0, 2]] -= dw
    #     boxes[:, [1, 3]] -= dh
    #     boxes = np.clip(boxes, 0, None).round().astype(np.int32)

    return  np.concatenate((boxes, scores.reshape(-1,1), labels.reshape(-1,1)), axis=1)



def scale_coords(boxes, ratio, pad, original_shape):
    oh, ow = original_shape[:2]
    dw, dh = pad
    
    # 去除填充
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    
    # 缩放回原始尺寸
    boxes[:, :4] /= ratio  # 假设宽高比例相同（r=width_scale=height_scale）
    
    # 裁剪到原始图像边界
    boxes[:, 0] = np.clip(boxes[:, 0], 0, ow)  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, oh)  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, ow)  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, oh)  # y2
    
    return boxes

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 80
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = self.engine.get_tensor_shape(self.engine.get_tensor_name(0))[2:]  # get the read shape of model, in case user input it wrong
        self.context = self.engine.create_execution_context()
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
                'size': size
            }
            self.allocations.append(allocation)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, img):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """

        # Prepare the output data.
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network.
        common.memcpy_host_to_device(self.inputs[0]['allocation'], np.ascontiguousarray(img))

        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            common.memcpy_device_to_host(outputs[o], self.outputs[o]['allocation'])
        return outputs

    def detect_video(self, video_path, conf=0.5, end2end=False):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('results.avi',fourcc,fps,(width,height))
        fps = 0
        import time
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            t1 = time.time()
            data = self.infer(blob)
            fps = (fps + (1. / (time.time() - t1))) / 2
            frame = cv2.putText(frame, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                dets = np.concatenate([np.array(final_boxes)[:int(num[0])], np.array(final_scores)[:int(num[0])], np.array(final_cls_inds)[:int(num[0])]], axis=-1)
            else:
                predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                dets = self.postprocess(predictions,ratio)

            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:,
                                                                :4], dets[:, 4], dets[:, 5]
                frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                                conf=conf, class_names=self.class_names)
            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    def inference(self, img_path, conf=0.5, end2end=False):
        origin_img = cv2.imread(img_path)
        # img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        img, ratio, dwdh, test_im = letterbox(origin_img, self.imgsz)
        data = self.infer(img)
        if end2end:
            num, final_boxes, final_scores, final_cls_inds  = data
            # final_boxes, final_scores, final_cls_inds  = data
            dwdh = np.asarray(dwdh * 2, dtype=np.float32)
            final_boxes -= dwdh
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            final_scores = np.reshape(final_scores, (-1, 1))
            final_cls_inds = np.reshape(final_cls_inds, (-1, 1))
            dets = np.concatenate([np.array(final_boxes)[:int(num[0])], np.array(final_scores)[:int(num[0])], np.array(final_cls_inds)[:int(num[0])]], axis=-1)
        else:
            # predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            # dets = self.postprocess(predictions, ratio, dwdh, origin_img.shape)
            anchors = [
            [[10, 13], [16, 30], [33, 23]],  # 小特征层（如80x80）
            [[30, 61], [62, 45], [59, 119]], # 中特征层（如40x40）
            [[116, 90], [156, 198], [373, 326]]  # 大特征层（如20x20）
            ]
            dets = yolov5_postprocess(    
                preds=data,
                anchors=anchors,
                img_size=640,
                conf_thres = conf,
                original_shape=origin_img.shape[:2],
                ratio = ratio,
                dwdh = dwdh  
                )
            

        # if len(dets[0]) > 0:
        #     final_boxes, final_scores, final_cls_inds = dets[:,
        #                                                      :4], dets[:, 4], dets[:, 5]
        #     origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
        #                      conf=conf, class_names=self.class_names)
        return origin_img, dets

    @staticmethod
    def postprocess(predictions, ratio, dwdh, origin_shape):
        boxes = predictions[:, :4]
        scores = predictions[:, 5:]
        conf = predictions[:, 4]
        # boxes = decode_yolo(boxes, anchors=[[10,13, 16,30, 33,23], 
        #                     [30,61, 62,45, 59,119], 
        #                     [116,90, 156,198, 373,326]],
        #            strides=[8, 16, 32])
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy = scale_coords(boxes_xyxy, ratio, dwdh, origin_shape)

        dets = multiclass_nms(boxes_xyxy, scores, conf, nms_thr=0.45, score_thr=0.25)
        return dets

    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, conf, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = conf > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def letterbox(im,
              new_shape = (640, 640),
              color = (114, 114, 114),
              swap=(2, 0, 1)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    test_im = copy.deepcopy(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.transpose(swap)
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.
    return im, r, (dw, dh), test_im


def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img