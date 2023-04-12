import os
import time
import argparse
from multiprocessing import Pool

parser_auto_cali = argparse.ArgumentParser(description='uto_cali params.', conflict_handler='resolve')
parser = argparse.ArgumentParser(description='auto_cali_test params.')
parser.add_argument('--debug_cmd', type=str, default='onnx,sym', help='exclude')
opt = parser.parse_args()

model_list_all=[
  # "--arch=shufflenet_v2_x0_5 --batch-size=320 --lr=1e-2",
  #"--arch=mobilenet_v2 --batch-size=128 --lr=1e-3", 
  "--arch=resnet18 --batch-size=256 --lr=1e-2",
  # "--arch=vgg11_bn --batch-size=32 --lr=1e-3",
  #"--arch=resnet50 --batch-size=32 --lr=1e-2",
  # "--arch=squeezenet1_1 --batch-size=128 --lr=1e-3",
  #"--arch=mobilenet_v3_small  --batch-size=128 --lr=1e-2"
]

epochs = 3
output_path='/workspace/tmp_path_0322_tmptest'
output_path='/workspace/tmp_path_0412_tmptest'
fast_test = ''
fast_test = '--fast_test'
pre_eval_and_export = '--pre_eval_and_export'
pre_eval_and_export = ''
cmd_str = f"--epochs={epochs} --deploy_batch_size=10 --gpu=0 --pretrained --evaluate --backend=sophgo_tpu --optim=sgd \
           --train_data=/workspace/for_train_val/ --val_data=/workspace/for_train_val/ --output_path={output_path} {fast_test} {pre_eval_and_export}"

def worker(cmd_line):
    print('cmd_line:', cmd_line)
    os.system(cmd_line)

if __name__ == "__main__":
  wp = os.getcwd()
  po = Pool(1)
  for i,model in enumerate(model_list_all):
    arch = model.split(' ')[0].split('=')[1].strip()
    os.system(f'mkdir -p {output_path}/{arch}')
    cmd_line = f'cd {wp};python3 application/imagenet_example/main.py  {model} {cmd_str} > {output_path}/{arch}/{i}_log_train_{arch} 2>&1'
    print('start', model)
    po.apply_async(worker, (cmd_line,))

  po.close()
  po.join()
  print('all end')
