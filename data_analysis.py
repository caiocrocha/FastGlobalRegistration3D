#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    result = pd.read_csv('test.csv').values.tolist()

    # Extract the noise level and test number from the first element in each list
    idx = [x[0].split("_rot_")[0].split("_")[-2:] for x in result]
    # Convert the noise level and test number to integers
    idx = [(int(x[0]), int(x[1])) if len(x[0]) == 2 else (0, int(x[1])) for x in idx]

    # Extract the time and RMSE from the second and third elements in each list
    result = np.array([[float(xi) for xi in x[1:]] for x in result])
    # Create a DataFrame with the extracted data
    data = pd.DataFrame(idx, columns=['noise', 'test'])
    data['fgr_rmse'] = result[:,0]
    data['fgr_time'] = result[:,1]
    data['ransac_rmse'] = result[:,2]
    data['ransac_time'] = result[:,3]

    print('Mean RMSE by noise level (FGR):')
    print(data.groupby('noise').mean()['fgr_rmse'])
    print('Mean RMSE by noise level (RANSAC):')
    print(data.groupby('noise').mean()['ransac_rmse'])

    # Visualize the RMSE by test number
    fig, ax = plt.subplots()
    data.groupby('noise').plot(kind='line', x='test', y='fgr_rmse', ax=ax, legend='noise', c='tab:blue')
    data.groupby('noise').plot(kind='line', x='test', y='ransac_rmse', ax=ax, legend='noise', c='tab:red')
    plt.title('RMSE by test number')
    plt.xlabel('Test Number')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

    # Reproduce Figure 3 from the paper

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

    for method in ['fgr', 'ransac']:
        alpha_recall = []
        stepsize = 0.0001
        alpha_arr = np.arange(0, 0.06 + stepsize, stepsize)

        # Group the data by noise level
        grouped_data = data.groupby('noise')[f'{method}_rmse'].apply(list).to_dict()

        for alpha in alpha_arr:
            recall = []
            # count the number of entries where RMSE < alpha for each noise level 
            for noise_level, rmse_values in grouped_data.items():
                count = sum(rmse < alpha for rmse in rmse_values)
                recall.append(count / len(rmse_values))
            alpha_recall.append(recall)

        # Convert alpha_recall to a DataFrame for easier plotting
        alpha_recall_df = pd.DataFrame(alpha_recall, columns=grouped_data.keys(), index=alpha_arr)

        # Plot for each noise level
        for i, noise_level in enumerate(sorted(grouped_data.keys())):
            ax[i].plot(alpha_arr, alpha_recall_df[noise_level], label=method.upper(), c='tab:red' if method == 'fgr' else 'tab:green')
    
    for i, noise_level in enumerate(sorted(grouped_data.keys())):
        ax[i].set_title(f'Noise Level {noise_level}')
        ax[i].set_xlabel('Alpha')
        ax[i].set_ylabel('Percentage')
        ax[i].legend()

    plt.tight_layout()
    plt.savefig('alpha_recall_1.png')
    plt.show()