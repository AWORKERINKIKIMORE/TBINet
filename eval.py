import argparse
import os.path as osp
import os
from utils.evaluator import Eval_thread
from utils.data import EvalDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(cfg):

    root_dir = cfg.root_dir
    gt_dir = cfg.gt_dir

    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = root_dir

    method_names = cfg.methods
    dataset_names = cfg.datasets

    threads = []

    for method in method_names:

        test_res = []

        for dataset in dataset_names:
            loader = EvalDataset(osp.join(root_dir, method, dataset), osp.join(gt_dir, dataset, 'GT'))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)

            ##

            print('Evaluating_______', dataset)
            print(thread.run())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    gt_path = './dataset/test/'
    sal_path = './test_maps/'
    test_datasets = ['NJU2K', 'NLPR']

    parser.add_argument('--methods',  type=str,  default=['TBINet'])
    parser.add_argument('--datasets', type=str,  default=test_datasets)
    parser.add_argument('--gt_dir',   type=str,  default=gt_path)
    parser.add_argument('--root_dir', type=str,  default=sal_path)
    parser.add_argument('--save_dir', type=str,  default=None)
    parser.add_argument('--cuda',     type=bool, default=True)
    cfg = parser.parse_args()
    main(cfg)
