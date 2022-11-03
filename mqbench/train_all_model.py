import os
import time
import argparse
from multiprocessing import Pool

parser_auto_cali = argparse.ArgumentParser(description='uto_cali params.', conflict_handler='resolve')
parser = argparse.ArgumentParser(description='auto_cali_test params.')
parser.add_argument('--debug_cmd', type=str, default='onnx,sym', help='exclude')
opt = parser.parse_args()

model_list_all=[
  #"--arch=shufflenet_v2_x0_5 --batch-size=320",
  "--arch=mobilenet_v2 --batch-size=64",
  "--arch=resnet18 --batch-size=128",
  "--arch=vgg11_bn --batch-size=32",
  "--arch=resnet50 --batch-size=32",
  #"--arch=squeezenet1_1 --batch-size=128",

  #"--arch=mobilenet_v3_small  --batch-size=128"
]

cmd_str = "--epochs=10 --lr=1e-4 --gpu=0 --pretrained --evaluate --backend=sophgo_tpu --optim=sgd --pre_eval_and_export --train_data=/data/imagenet/for_train_val/ --val_data=/data/imagenet/for_train_val/ --output_path=/workspace/tmp_path_1024"#  --fast_test"

def worker(cmd_line):
    os.system(cmd_line)

if __name__ == "__main__":
  wp = os.getcwd()
  po = Pool(1)
  for i,model in enumerate(model_list_all):
    cmd_line = 'cd {};python3 imagenet_example/main.py  {} {} >{}_log_train_{} 2>&1'.format(wp, model, cmd_str, i, model.split(' ')[0].split('=')[1])
    print('start', model)
    po.apply_async(worker, (cmd_line,))

  po.close()
  po.join()
  print('all end')
