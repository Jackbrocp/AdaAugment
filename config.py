import argparse

def get_arg():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpus', type=str, default='2,3')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--log_interval',type=int, default=50, help='log training status')

    # Ooptimizer & target network parameters
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--conf', default='./confs/resnet18.yaml', type=str, help='yaml file')
    
    # Dataset parameters
    parser.add_argument('--cutout_length', type=int, default=16)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_worker', type=int, default=8, choices=[2, 4, 8, 16, 32])
    parser.add_argument('--aug', type=str, default='trivialaugment', choices=['autoaugment','randaugment','trivialaugment'])
    
    # A2C parameters
    parser.add_argument('--use_reward_norm',action='store_true')
    parser.add_argument('--use_orthogonal_init', action='store_true')
    parser.add_argument('--action_dim', type=int, default=1)
    parser.add_argument('--state_dim', type=int, default=512)
    
    args = parser.parse_args()
    return args