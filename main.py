import argparse
from src.trainer import trainer

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_iters',
        type=int,
        default=150000,
        help='The maximum iteration of training loop'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['phototourism', 'llff'],
        default='phototourism',
        help='The type of dataset'
    )

    parser.add_argument(
        '--scene',
        type=str,
        choices=['sacre', 'trevi', 'bandenburg'],
        default='sacre',
        help='The type of scene'
    )
    parser.add_argument(
        '--inner_step_size',
        type=int,
        default=1,
        help='The fewshot learning step size'
    )

    parser.add_argument(
        '--inner_update_steps',
        type=int,
        default=64,
        help='The fewshot learning update steps'
    )

    parser.add_argument(
        '--test_inner_steps',
        type=int,
        default=64,
        help='The test time fewshot learning update steps'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='The batch size in training loop'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=5e-4,
        help='The learning rate in training loop'
    )

    parser.add_argument(
        '--N_samples',
        type=int,
        default=128,
        help='The sampling point numbers in the ray'
    )

    parser.add_argument(
        '--datadir',
        type=str,
        default=f'/mnt/hdd1/stella/inerf/learnit_Data',
        help='The data loading base path'
    )

    args = parser.parse_args()

    # args.posedir = args.datadir + f'/phototourism/sacre/'  # Directory condtains [bds.npy, c2w_mats.npy, kinv_mats.npy, res_mats.npy]
    # args.imgdir = args.datadir + f'/phototourism/original/sacre/sacre_coeur/dense/images/'  # Directory of images
    args.posedir = args.datadir + f'/phototourism/sacre/'  # Directory condtains [bds.npy, c2w_mats.npy, kinv_mats.npy, res_mats.npy]
    args.imgdir = args.datadir + f'pull-phototourism-images/sacre_coeur/dense/images/'  # Directory of images
    
    my_trainer = trainer(args)
    my_trainer.train()