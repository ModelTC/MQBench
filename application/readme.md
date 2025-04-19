# cls_example
```shell
# train pytorch model
python train.py --data <your_dataset> --checkpoint_dir <your_save_path_torch>
# finetune qat model with pytorch model 
python train_qat.py --data <your_dataset> --checkpoint-dir <your_save_path_qat> --load-path <your_save_path_torch>
# convert your qat model
python convert.py --model-path <your_save_path_torch> --onnx-path <your_converted_model>
# eavl onnx
python eval_onnx.py --model-path <your_onnx_model> --data-dir <your_dataset>
# eval tensorrt
python trt_test.py --onnx-path <your_onnx_model> --trt-path <your_save_path_trt> --evaluate --data-path <your_dataset>
```
w\a Obsever MSE  
w/a Fakequantize Learnable​  
epoch 5

| epoch | lr   | Acc@1  | Acc@5  | PTQ acc              |
| ---- | -------- | ------ | ------ | ------------------------- |
| 1    | 0.00075  | 73.38  | 91.10  | Acc@1 72.13<br>Acc@5 90.68 |
| 2    | 0.000025 | 73.35  | 91.32  | -                         |
| 3    | 0      | 73.93  | 91.49  | -                         |



# yolov5_example
yolov5 https://github.com/ultralytics/yolov5/tree/f4d8a84c3855fcad94bdb1104c48d3804adc7b10
TensorRT-For-YOLO-Series https://github.com/Linaom1214/TensorRT-For-YOLO-Series/commit/34aa8ba1036b93f7602c9bf9051adbd66b67a095
```shell
# train qat model
cd yolov5
python train.py --data <coco_path> --epochs 5 --weights yolov5s.pt --cfg yolov5s.yaml --hyp data/hyps/hyp.no-augmentation.yaml --batch-size 64
# test qat model
cd ../TensorRT-For-YOLO-Series
python val.py -e <your_trt_path> --coco_annotation <your_coco_annotation> --coco_image_dir <your_coco_image_dir>
```
w\a Obsever MSE  
w/a Fakequantize Learnable​  
epoch 5
yolov5s
| model | mAP |
| ---- | ------------------------- |
| qat 5   | 0.284 |
| ptq     | 0.277  |
| fp32    | 0.287  |