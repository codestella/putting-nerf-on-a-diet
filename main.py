import argparse
from src.trainer import Trainer

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
        choices=['sacre', 'trevi', 'bandenburg', 'notre'],
        default='notre',
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
        default='/content/gdrive/MyDrive/Colab_codes/',
        help='The data loading base path'
    )

    parser.add_argument(
        '--select_data',
        type=str,
        default='phototourism/notre',
        help="Select data to use e.g.) 'nerf_synthetic/lego', 'phototourism/sacre', 'shapenet/chair'"
    )

    parser.add_argument(
        '--pretrained',
        type=str,
        help="load pre-trained parameters in the google drive; learnit_data"
    )

    args = parser.parse_args()
    
    my_trainer = Trainer(args)
    my_trainer.train()