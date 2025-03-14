#!/usr/bin/env python3

# encoding: utf-8

import open3d as o3d
import numpy as np
import time
import copy
import os

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                    #   zoom=0.4559,
                                    #   front=[0.6452, -0.3036, -0.7011],
                                    #   lookat=[1.9892, 2.0208, 1.8945],
                                    #   up=[-0.2779, -0.9482 ,0.1556]
                                      )

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = pcd

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source_pth, target_pth):
    # print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
    # source = o3d.io.read_point_cloud("data/cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud("data/cloud_bin_1.pcd")
    # source = o3d.io.read_point_cloud("data/bunny_perturbed.ply")
    # target = o3d.io.read_point_cloud("data/bunny_original.ply")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    source = o3d.io.read_point_cloud(source_pth)
    target = o3d.io.read_point_cloud(target_pth)
    # source = o3d.io.read_point_cloud("data/features_0000.bin")
    # target = o3d.io.read_point_cloud("data/features_0001.bin")

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    mutual_filter = False
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
    #         % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

voxel_size = 0.05 # means 5cm for this dataset

print("filename,time(s),rmse")
for dire in os.listdir("FastGlobalRegistration/dataset"):
    if os.path.isdir("FastGlobalRegistration/dataset/" + dire):
        source_pth = "FastGlobalRegistration/dataset/" + dire + "/Depth_0000.ply"
        target_pth = "FastGlobalRegistration/dataset/" + dire + "/Depth_0001.ply"

        source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(voxel_size, source_pth, target_pth)

        # start = time.time()
        # result_ransac = execute_global_registration(source_down, target_down,
        #                                             source_fpfh, target_fpfh,
        #                                             voxel_size)
        # print("Global registration took %.3f sec.\n" % (time.time() - start))
        # print(result_ransac)
        # draw_registration_result(source_down, target_down,
        #                          result_ransac.transformation)

        start = time.time()
        result_fast = execute_fast_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        # print("Fast global registration took %.3f sec.\n" % (time.time() - start))
        print(f"{dire},{time.time() - start},{result_fast.inlier_rmse}")
        # draw_registration_result(source_down, target_down,
        #                          result_fast.transformation)

        # ransac_T = source.transform(result_ransac.transformation)
        # o3d.io.write_point_cloud("RANSAC.ply", ransac_T)

        fast_T = source.transform(result_fast.transformation)
        o3d.io.write_point_cloud("FastGlobalRegistration.ply", fast_T)

        # result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
        #                                  voxel_size)
        # print(result_icp)
        # draw_registration_result(source, target, result_icp.transformation)