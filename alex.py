import argparse
import os
import time

import pandas as pd
from tqdm import tqdm

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Main script to reproduce results on the Synthetic Range and Scene datasets")

    parser.add_argument('--data_path',
                        type=str,
                        default='./data/',
                        help='Path to the datasets')

    args = parser.parse_args()

    synth_path = args.data_path + 'synthetic/'
    scene_path = args.data_path + 'scene/'

    # Synthetic Range Dataset
    synth_results = []

    voxel_size = 0.05

    for d in tqdm(os.listdir(synth_path), desc='Processing Synthetic Range dataset'):
        source_path = synth_path + d + '/Depth_0000.ply'
        target_path = synth_path + d + '/Depth_0001.ply'

        source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(voxel_size, source_path, target_path)

        fgr_start = time.time()
        result_fgr = execute_fast_global_registration(
            source_down, target_down,
            source_fpfh, target_fpfh,
            voxel_size
        )

        ransac_start = time.time()
        result_ransac = execute_global_registration(
            source_down, target_down,
            source_fpfh, target_fpfh,
            voxel_size
        )

        synth_results.append(
            {
                'file': d,
                'fgr_rmse': result_fgr.inlier_rmse,
                'fgr_time': ransac_start - fgr_start,
                'ransac_rmse': result_ransac.inlier_rmse,
                'ransac_time': time.time() - ransac_start
            }
        )

    synth_df = pd.DataFrame(synth_results)
    synth_df.to_csv('test.csv', index=False)
    
    # Scene Dataset
    # for file in os.listdir("./data/range"):

    # Outliers