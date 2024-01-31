import open3d as o3d
import argparse
import numpy as np
import mrob
import math

from pathlib import Path
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

def get_intermediate_transform(T1, T2, t):
    dxi = mrob.geometry.SE3(T2 @ np.linalg.inv(T1)).Ln()
    return mrob.geometry.SE3(t * dxi).T() @ T1

def undistort_segment(pcd, theta_min, theta_max, color, T):
    # Convert the point cloud points to a numpy array
    points = np.asarray(pcd.points)
    thetas = np.asarray([math.atan2(p[1],p[0]) for p in points])

    # Create a mask for points that fall within the ROI
    mask_roi = (thetas < theta_max) & (thetas >= theta_min)
    trues = mask_roi.sum()

    new_pcd = o3d.geometry.PointCloud()
    if trues > 0:
        new_colors = np.full((trues, 3), color)
        new_points = points[mask_roi]

        new_pcd.points = o3d.utility.Vector3dVector(new_points)
        new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    return new_pcd.transform(T)

def undistort_cloud(open3d_cloud, color, N, T1, T2):
    N += 1 # starting separations from 0
    undistort_result = []
    prev_sector = -np.pi
    coefs = np.linspace(0, 1, N)
    sectors = np.linspace(-np.pi, np.pi, N)

    for i in range(1, N):
        T = get_intermediate_transform(T1, T2, coefs[i])
        color[2] = coefs[i]
        undistort_result.append(undistort_segment(open3d_cloud, prev_sector, sectors[i], color, np.asarray(T)))
        prev_sector = sectors[i]
    return undistort_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default='pointcloud_undistort/dataset',
        help="path to the pointclouds",
    )


    args = parser.parse_args()
    dataset = Path(args.dataset)

    path0 = str(dataset/"0.pcd")
    path9 = str(dataset/"9.pcd")

    pcd_start = o3d.io.read_point_cloud(path0)
    pcd_start = pcd_start.paint_uniform_color([1.0, 0.0, 0.0])

    pcd_fin = o3d.io.read_point_cloud(path9)
    pcd_fin = pcd_fin.paint_uniform_color([0.0, 1.0, 0.0])

    pcd_undistort = o3d.io.read_point_cloud(path9)
    pcd_undistort = pcd_undistort.paint_uniform_color([0.0, 0.0, 0.0])

    
    pose_start = np.eye(4, 4)
    pose_fin = np.asarray([
        [ 0.99832152,  0.02603589,  0.05173264,  0.05743424],
        [-0.02286862,  0.99788188, -0.0608998,  -0.04333022],
        [-0.05320864,  0.05961452,  0.99680236, -0.01237881],
        [ 0.0,         0.0,         0.0,         1.0     ]],
        )

    N_patitions = 16
    undistort_result = undistort_cloud(pcd_undistort, [0.0, 0.0, 0.0], N_patitions, pose_start, pose_fin)
    o3d.visualization.draw_geometries(undistort_result)
    o3d.visualization.draw_geometries([pcd_start,  pcd_fin] + undistort_result)
