import open3d as o3d


def preprocess_point_cloud(pcd, voxel_size):
    # Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # FPFH Features
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source_pth, target_pth):
    # Load Point Clouds
    source = o3d.io.read_point_cloud(source_pth)
    target = o3d.io.read_point_cloud(target_pth)

    # Compute FPFH Features and Normals
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    # Open3D's implementation for the FGR pairwise algorithm.
    
    # delta
    distance_threshold = voxel_size * 0.5
    
    # tau, ok but paper reports with 0.9
    edge_prune_threshold = 0.95

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold,
            tuple_scale=edge_prune_threshold))
    
    return result


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    # Open3D's implementation of RANSAC-based registration

    distance_threshold = voxel_size * 1.5
    mutual_filter = False

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
    return result