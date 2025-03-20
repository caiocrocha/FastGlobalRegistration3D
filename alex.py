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
    
    parser.add_argument('--synth',
                        action='store_true',
                        default=True,
                        help='Run the benchmark on the Synthetic Range dataset')
    
    parser.add_argument('--scene',
                        action='store_true',
                        default=True,
                        help='Run the benchmark on the Scene dataset')

    args = parser.parse_args()

    synth_path = args.data_path + 'synthetic/'
    scene_path = args.data_path + 'scene/'

    # Synthetic Range Dataset
    if args.synth == True:
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
        synth_df.to_csv('synth.csv', index=False)
    
    # Scene Dataset
    if args.scene == True:
        scene_results = []

        voxel_size = 0.05

        # for d in tqdm(os.listdir(scene_path), desc='Processing Scene dataset'):
        for d in ['livingroom1-fragments-ply']:
            file_results = []
            # files = [file for file in os.listdir(scene_path + d) if file[-4:] == '.ply']

            indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 42, 56]
            files = ['cloud_bin_' + str(idx) + '.ply' for idx in indices]

            for i in tqdm(range(len(files))):
                i_results = []
                
                source_path = scene_path + d + '/' + files[i]
                for j in tqdm(range(i + 1, len(files))):
                    
                    target_path = scene_path + d + '/' + files[j]

                    source, target, source_down, target_down, source_fpfh, target_fpfh = \
                        prepare_dataset(voxel_size, source_path, target_path)

                    result_fgr = execute_fast_global_registration(
                        source_down, target_down,
                        source_fpfh, target_fpfh,
                        voxel_size
                    )

                    fitness = result_fgr.fitness

                    if fitness < 0.3:
                        continue

                    i_results.append([i, j, fitness, result_fgr.transformation])

                file_results += i_results




        



    #     source_path = scene_path + d + '/Depth_0000.ply'
    #     target_path = scene_path + d + '/Depth_0001.ply'

    #     source, target, source_down, target_down, source_fpfh, target_fpfh = \
    #         prepare_dataset(voxel_size, source_path, target_path)

    #     fgr_start = time.time()
    #     result_fgr = execute_fast_global_registration(
    #         source_down, target_down,
    #         source_fpfh, target_fpfh,
    #         voxel_size
    #     )

    #     ransac_start = time.time()
    #     result_ransac = execute_global_registration(
    #         source_down, target_down,
    #         source_fpfh, target_fpfh,
    #         voxel_size
    #     )

    #     synth_results.append(
    #         {
    #             'file': d,
    #             'fgr_rmse': result_fgr.inlier_rmse,
    #             'fgr_time': ransac_start - fgr_start,
    #             'ransac_rmse': result_ransac.inlier_rmse,
    #             'ransac_time': time.time() - ransac_start
    #         }
    #     )

    # synth_df = pd.DataFrame(synth_results)
    # synth_df.to_csv('test.csv', index=False)
    # for file in os.listdir("./data/range"):

    # Hyperparameter Search


    # Outliers