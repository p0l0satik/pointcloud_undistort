import open3d as o3d
import argparse
import numpy as np
import mrob
import math

from pathlib import Path
from scipy.spatial.transform import Rotation as R


def get_intermediate_transform(initial_pose, target_pose, coef):
    dxi = mrob.geometry.SE3(target_pose @ np.linalg.inv(initial_pose)).Ln()
    return np.linalg.inv(mrob.geometry.SE3(coef * dxi).T() @ initial_pose)


def cut_segment(pcd, theta_min, theta_max, coef):
    # Convert the point cloud points to a numpy array
    points = np.asarray(pcd.points)
    thetas = np.asarray([math.atan2(p[1],p[0]) for p in points])

    # Create a mask for points that fall within the ROI
    mask_roi = (thetas < theta_max) & (thetas >= theta_min)
    trues = mask_roi.sum()

    new_pcd = o3d.geometry.PointCloud()
    if trues > 0:
        new_colors = np.full((trues, 3), [0., 0., coef])
        new_points = points[mask_roi]

        new_pcd.points = o3d.utility.Vector3dVector(new_points)
        new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    return new_pcd


def undistort_cloud(cloud, start_angle, N_segments, initial_pose, target_pose):
    N_segments += 1 # starting separations from 0
    undistort_result = []
    coefs = np.linspace(0, 1, N_segments)
    sectors = np.linspace(start_angle, start_angle - 2 * np.pi, N_segments)

    coefs = np.roll(coefs, -len(coefs)//2 )

    for i in range(1, N_segments):
        transform_matrix = get_intermediate_transform(initial_pose, target_pose, coefs[i-1])
        undistort_result.append(cut_segment(cloud, sectors[i], start_angle,  coefs[i-1]).transform(transform_matrix))
        start_angle = sectors[i]
    return undistort_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pcd_start",
        type=str,
        default='dataset/0.pcd',
        help="pcd of initial pose",
    )
    parser.add_argument(
        "--pcd_final",
        type=str,
        default='dataset/9.pcd',
        help="pcd of final pose",
    )
    parser.add_argument(
        "--N_segments",
        type=int,
        default=4,
        help="number of segments to split pointcloud into, must be an even number",
    )


    args = parser.parse_args()

    pcd_start = o3d.io.read_point_cloud(str(Path(args.pcd_start)))
    pcd_start = pcd_start.paint_uniform_color([1.0, 0.0, 0.0])

    pcd_fin = o3d.io.read_point_cloud(str(Path(args.pcd_final)))
    pcd_fin = pcd_fin.paint_uniform_color([0.0, 1.0, 0.0])

    pcd_undistort = o3d.io.read_point_cloud(str(Path(args.pcd_final)))
    pcd_undistort = pcd_undistort.paint_uniform_color([0.0, 0.0, 0.0])

    
    pose_start = np.eye(4, 4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_start, pcd_fin, 0.1, pose_start,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),  
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )

    start_angle = np.pi
    undistort_result = undistort_cloud(pcd_undistort, start_angle, args.N_segments, pose_start, reg_p2p.transformation)

    o3d.visualization.draw_geometries(undistort_result)
    o3d.visualization.draw_geometries([pcd_start,  pcd_fin] + undistort_result)
