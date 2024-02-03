import open3d as o3d
import argparse
import numpy as np
import mrob
import math

from pathlib import Path
from scipy.spatial.transform import Rotation as R


def get_intermediate_transform(initial_pose: np.ndarray, target_pose: np.ndarray, coef: float) -> np.ndarray:
    dxi = mrob.geometry.SE3(target_pose @ np.linalg.inv(initial_pose)).Ln()
    return mrob.geometry.SE3(coef * dxi).T() @ initial_pose


def cut_segment(pcd: o3d.geometry.PointCloud, theta_min: float, theta_max: float, coef: float) -> o3d.geometry.PointCloud:
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


def undistort_cloud(pcd_initial: o3d.geometry.PointCloud, pcd_target: o3d.geometry.PointCloud, initial_pose: np.ndarray, target_pose: np.ndarray, N_segments: int) -> o3d.geometry.PointCloud:
    # preventing objects from mutating
    pcd_initial = o3d.geometry.PointCloud(pcd_initial)
    pcd_target =  o3d.geometry.PointCloud(pcd_target)

    N_segments += 1 # starting separations from 0
    undistort_result = []
    coefs = np.linspace(0, 1, N_segments)
    start_angle = np.pi 
    sectors = np.linspace(start_angle, start_angle - 2 * np.pi, N_segments)

    # applying poses to pcd
    pcd_initial.transform(initial_pose)
    pcd_target.transform(target_pose)
    
    # calculating transform between two poses
    initial_transform = np.eye(4, 4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_initial, pcd_target, 0.1, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),  
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000),
    )

    # returning initial pcd to it's original state
    # this is done for correct sector cutting
    pcd_initial.transform(np.linalg.inv(initial_pose))


    coefs = np.roll(coefs, -len(coefs)//2 ) # rolled due to lidar ray start
    for i in range(1, N_segments):
        # cutting a segment
        segment = cut_segment(pcd_initial, sectors[i], start_angle,  coefs[i-1])

        # getting composite transform matrix: initial -> pose --(gradual)--> target_pcd pose
        transform_matrix = get_intermediate_transform(initial_pose, reg_p2p.transformation@initial_pose, coefs[i-1])

        # applying transform to the segment
        undistort_result.append(segment.transform(transform_matrix))
        
        start_angle = sectors[i]

    # combining segments in one pcd
    return_pcd = o3d.geometry.PointCloud()
    return_pcd_points = return_pcd.points
    return_pcd_colors = return_pcd.colors

    for segment_pcd in undistort_result:
        return_pcd_points.extend(segment_pcd.points)
        return_pcd_colors.extend(segment_pcd.colors)

    return_pcd.points =  return_pcd_points  
    return_pcd.colors = return_pcd_colors

    return return_pcd


def read_pose(filename: str):
    matrix = np.eye(4)
    with open(filename, newline='') as f:
        lines = f.readlines()
        matrix = np.asarray([[round(float(el), 4) for el in row.split(" ")[0:7]] for row in lines])
        
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pcd_initial",
        type=str,
        default='/home/polosatik/pointcloud_undistort/hilti_14/clouds/0.pcd',
        help="pcd of initial pose",
    )
    parser.add_argument(
        "--pcd_target",
        type=str,
        default='/home/polosatik/pointcloud_undistort/hilti_14/clouds/15.pcd',
        help="pcd of final pose",
    )

    parser.add_argument(
        "--initial_pose",
        type=str,
        default='/home/polosatik/pointcloud_undistort/hilti_14/poses/0.txt',
        help="pose of initial pose",
    )
    parser.add_argument(
        "--target_pose",
        type=str,
        default='/home/polosatik/pointcloud_undistort/hilti_14/poses/15.txt',
        help="pose of final pose",
    )

    parser.add_argument(
        "--N_segments",
        type=int,
        default=8,
        help="number of segments to split pointcloud into, must be an even number",
    )


    args = parser.parse_args()

    pcd_initial = o3d.io.read_point_cloud(str(Path(args.pcd_initial)))
    pcd_initial = pcd_initial.paint_uniform_color([1.0, 0.0, 0.0])

    pcd_target = o3d.io.read_point_cloud(str(Path(args.pcd_target)))
    pcd_target = pcd_target.paint_uniform_color([0.0, 1.0, 0.0])

    pcd_undistort = o3d.io.read_point_cloud(str(Path(args.pcd_initial)))

    initial_pose = read_pose(Path(args.initial_pose))
    target_pose = read_pose(Path(args.target_pose))

    undistort_pcd = undistort_cloud(pcd_initial, pcd_target, initial_pose, target_pose, args.N_segments)
    
    pcd_initial.transform(initial_pose)
    pcd_target.transform(target_pose)
    o3d.visualization.draw_geometries([pcd_initial,  pcd_target, undistort_pcd])
