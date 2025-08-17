import argparse
import os
import json
from util.datautils import load_UCR

parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--save_path', type=str, default='FreConvNet/test')
parser.add_argument('--data_path', type=str,
                    default='/data_provider/')
parser.add_argument('--dataset_name', type=str, default='Basicmotions')
parser.add_argument('--device', type=str, default='cuda:0')#cuda:0
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)

# model args
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--loss', type=str, default='ce', choices=['bce', 'ce'])
parser.add_argument('--save_model', type=int, default=1)
#FreConvNet
parser.add_argument('--patch_method', type=str, default='CI',choices=['CF', 'CI',], help='different patch methods')
parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
parser.add_argument('--ffn_ratio', type=int, default=4, help='ffn_ratio')
parser.add_argument('--patch_size', type=int, default=1, help='the patch size')
parser.add_argument('--patch_stride', type=int, default=1, help='the patch stride')
parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1], help='num_blocks in each stage')
parser.add_argument('--num_stage', nargs='+',type=int, default=2, help='number of stage')
parser.add_argument('--dims', nargs='+',type=int, default=[64,64,64], help='dmodels in each stage')
# train args
parser.add_argument('--lr', type=float, default=0.015)
parser.add_argument('--lr_decay_rate', type=float, default=1.)
parser.add_argument('--lr_decay_steps', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--train_optim', type=str, default='AdamW',choices={'AdamW', 'Adam', 'RAdam'})

#multi task
parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, choices:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')

args = parser.parse_args()

if args.data_path is None:
    Train_data, Test_data = load_UCR(folder=args.dataset_name)
    args.num_class = len(set(Train_data[1]))
else:
    path = args.data_path
    Train_data, Test_data = load_UCR(path, folder=args.dataset_name)
    args.num_class = len(set(Train_data[1]))

args.eval_per_steps = max(1, int(len(Train_data[0]) / args.train_batch_size))
args.lr_decay_steps = args.eval_per_steps
save_path = args.save_path+'/'+args.dataset_name
if not os.path.exists(save_path):
    os.makedirs(save_path)
config_file = open(save_path +'/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
