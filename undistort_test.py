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

def color_segment(pcd, theta_min, theta_max, color):
    # Convert the point cloud points to a numpy array
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    thetas = np.asarray([math.atan2(p[1],p[0]) for p in points])
    # Create a mask for points that fall within the ROI
    mask_roi = (thetas < theta_max) & (thetas >= theta_min)
    trues = mask_roi.sum()
    if trues > 0:
        colors[mask_roi] =  np.full((trues, 3), color)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def color_cloud(open3d_cloud, color):
    prev_sector = -np.pi
    steps = 10
    for sector in np.linspace(-np.pi, np.pi, steps):
        if sector == -np.pi:
            continue
        open3d_cloud = color_segment(open3d_cloud, prev_sector, sector, color)
        prev_sector = sector
        color[2]+= 1.0 / steps
    return open3d_cloud

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
    pcds = []
    prev_sector = -np.pi
    coefs = np.linspace(0, 1, N)
    sectors = np.linspace(-np.pi, np.pi, N)
    print(sectors)
    for i in range(1, N):
        T = get_intermediate_transform(T1, T2, coefs[i])
        color[2] = coefs[i-1]
        pcds.append(undistort_segment(open3d_cloud, prev_sector, sectors[i], color, np.asarray(T)))
        prev_sector = sectors[i]
    return pcds


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

    pose0 = o3d.io.read_point_cloud(path0)
    pose0 = pose0.paint_uniform_color([1.0, 0.0, 0.0])

    pose9 = o3d.io.read_point_cloud(path9)
    pose9 = pose9.paint_uniform_color([0.0, 1.0, 0.0])

    pose2 = o3d.io.read_point_cloud(path9)
    pose2 = pose2.paint_uniform_color([0.0, 0.0, 0.0])

    pose3 = o3d.io.read_point_cloud(path9)
    pose3 = pose3.paint_uniform_color([1.0, 1.0, 0.0])

    
    trans_init = np.eye(4, 4)
    pose_fin = np.asarray([
        [ 0.99832152,  0.02603589,  0.05173264,  0.05743424],
        [-0.02286862,  0.99788188, -0.0608998,  -0.04333022],
        [-0.05320864,  0.05961452,  0.99680236, -0.01237881],
        [ 0.0,          0.0,          0.0,          1.0        ]],
        )


    pcds = undistort_cloud(pose2, [0.0, 0.0, 0.0], 16, trans_init, pose_fin)
    o3d.visualization.draw_geometries(pcds)
    o3d.visualization.draw_geometries([pose0,  pose9] + pcds)
