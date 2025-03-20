import os

import numpy as np
from scipy.spatial.transform import Rotation as R

import open3d as o3d
from tqdm import tqdm

def load_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)

    # Downsampling to 1000 points
    n = 1000
    voxel_size = 0.05
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    while len(np.array(downsampled_pcd.points)) != n:
        if len(np.array(downsampled_pcd.points)) > n:
            voxel_size *= 1.2
        else:
            voxel_size *= 0.8

        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    # Resizing to fit in [0, 1]^3
    points = np.array(downsampled_pcd.points)

    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    scale = np.min(1 / (max_vals - min_vals))

    return (points - min_vals)*scale

def save_ply(cloud, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    o3d.io.write_point_cloud(path, pcd)

def rotate(n):
    rotation_matrices = []

    for _ in range(n):
        angles = np.random.uniform(0, 360, 3)
        rotation_matrices.append(R.from_euler('xyz', angles, degrees=True).as_matrix())
    
    return np.array(rotation_matrices)

def scale(n):
    # Scaling factor from  1 to 5
    return np.random.uniform(0, 5, n)

def translate(n):
    translation_vectors = np.random.uniform(-1, 1, (n, 3))

    # Ensure that each vector has at most norm 1
    return translation_vectors / np.linalg.norm(translation_vectors, axis=1, keepdims=True) * np.random.uniform(0, 1)

def noise(n):
    noise_vectors = []

    # Noise sampled from a Gaussian distribution of zero mean and standard deviation 0.01
    # Values clipped to a beta parameter set at 5.54 the standard deviation.
    sigma = 0.01
    beta = 5.54*sigma

    noise_vectors = np.random.normal(0, sigma, (n, 3))

    return noise_vectors / np.linalg.norm(noise_vectors, axis=1, keepdims=True) * np.random.uniform(0, beta)

def outliers(cloud, p):
    n = int(p*cloud.shape[0])

    indices = np.random.randint(cloud.shape[0], size=n)

    theta = np.random.uniform(0, 2*np.pi, n)
    phi = np.random.uniform(0, np.pi, n)
    r = 5*(np.random.uniform(0,1, n) ** (1/3)) # Radius 5

    outliers = np.stack((r*np.sin(phi)*np.cos(theta),
                         r*np.sin(phi)*np.sin(theta),
                         r*np.cos(phi)), axis=-1)
    
    cloud[indices] = outliers

    return cloud

outliers_path = './data/outliers/'

if not os.path.isdir(outliers_path):
    os.makedirs(outliers_path)

    bunny_path = './data/bunny_original.ply'

    bunny = load_ply(bunny_path)

    save_ply(bunny, './data/bunny_d.ply')

    n = 5

    rotations = rotate(n)
    scales = scale(n)
    translations = translate(n)
    noises = noise(n)

    pbar = tqdm(total=10*n**4, desc="Generating Outlier Dataset", dynamic_ncols=True)
    for o in range(0, 100, 10):
        os.makedirs(outliers_path + str(o))
        
        i = 0

        for r in rotations:
            for s in scales:
                for t in translations:
                    for e in noises:
                        transformed_bunny = s*np.dot(bunny, r.T) + t + e

                        save_ply(transformed_bunny, outliers_path + str(o) + '/' + str(i) + '.ply')
                        
                        transformation_matrix = np.eye(4)
                        transformation_matrix[:3, :3] = s*r
                        transformation_matrix[:3, 3] = t + e

                        with open(outliers_path + str(o) + '/' + str(i) + '.log', 'w') as f:
                            f.write("0 1 2\n")
                            for row in transformation_matrix:
                                formatted_row = " ".join(f"{value:.10f}" for value in row)
                                f.write(f"{formatted_row}\n")

                        i += 1
                        pbar.update(1)


# n**4*(220 B + )
# create a folder named outliers
# for i in range(0, 100, 10)
# create folder 




