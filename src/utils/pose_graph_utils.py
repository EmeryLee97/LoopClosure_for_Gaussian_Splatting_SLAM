import random
from typing import List, Union, Tuple

import numpy as np
import torch
import open3d as o3d
import theseus as th
from scipy.spatial import KDTree


from src.utils.utils import torch2np
from src.utils.gaussian_model_utils import build_scaling_rotation, build_rotation


def downsample(container, num_samples: int):
    """ downsample elements inside a container with a given number """
    num_before = len(container)
    print(f"Trying to downsample {num_samples} points.")
    if num_before <= num_samples:
        return list(range(0, num_before))
    else:
        return random.sample(range(0, num_before), num_samples)


def match_gaussian_means(pts_1: torch.Tensor, pts_2: torch.Tensor, transformation: torch.Tensor, epsilon=5e-2) -> Tuple[List]:
    """ Select inlier correspondences from two Gaussian clouds, use kd-tree to speed up """
    print(f"pts_1 is at {pts_1.device}")
    print(f"transformation matrix is at {transformation.device}")
    rotation = transformation[:3, :3].to(pts_1.device)
    translation = transformation[:3, 3].to(pts_1.device)
    pts_1_new = pts_1 @ rotation.transpose(-1, -2) + translation
    pts2_kdtree = KDTree(torch2np(pts_2))

    _, query_idx = pts2_kdtree.query(
        torch2np(pts_1_new), 
        distance_upper_bound=epsilon, 
        workers=-1
    )
    res_list_1, res_list_2 = [], []
    for i in range(query_idx.shape[0]):
        if query_idx[i] != pts_2.shape[0]:
            res_list_1.append(i)
            res_list_2.append(query_idx[i])
    print(f'{len(res_list_1)} correspondences are found before adding to pose graph.')
    return res_list_1, res_list_2


def modified_sigmoid(tensor: torch.Tensor, k=5) -> torch.Tensor:
    """ A modified version of sigmoid function which only accepts non-negative inputs, and return values are limited
    between 0 and 1. Modify k to get different curvatures 
    """
    return 2.0 / (1 + torch.exp(k * tensor))


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    return L @ L.transpose(-2, -1)


def hellinger_distance(
    gaussian_xyz_i: torch.Tensor, gaussian_covariance_i: torch.Tensor,
    gaussian_xyz_j: torch.Tensor, gaussian_covariance_j: torch.Tensor,
) -> torch.Tensor:
    """ Computes the squared Hellinger distance between two gaussian distributions, batch operation supported 
    Args:
        gaussian_xyz_i, gaussian_xyz_j: gaussian means, shape = [..., 3]
        gaussian_scaling_i, gaussian_scaling_j: scaling part of covariance matrix, represented as 3d vectors, shape = [..., 3]
        gaussian_rotation_i, gaussian_rotation_j: rotation part of covariance matrix, represented as quaternions, shape = [..., 4]
    Returns:
        Squared Hellinger distance of two Gaussian distributions, limited in [0, 1]
    """
    gaussian_covariance_mean = (gaussian_covariance_i + gaussian_covariance_j) / 2 
    # det_gaussian_covariance_i = torch.linalg.det(gaussian_covariance_i)
    # det_gaussian_covariance_j = torch.linalg.det(gaussian_covariance_j)
    # det_gaussian_covariance_mean = torch.linalg.det(gaussian_covariance_mean)
    gaussian_xyz_diff = (gaussian_xyz_i - gaussian_xyz_j).unsqueeze(-1)
    # coefficient = det_gaussian_covariance_i.pow(1/4) * det_gaussian_covariance_j.pow(1/4) / det_gaussian_covariance_mean.pow(1/2)
    power = -1/8 * gaussian_xyz_diff.transpose(-2, -1) @ gaussian_covariance_mean.inverse() @ gaussian_xyz_diff
    # h_distance = 1 - coefficient * torch.exp(power.squeeze())
    h_distance = 1 - torch.exp(power.squeeze())
    return h_distance


def error_fn_dense_gaussian_alignment(optim_vars, aux_vars) -> torch.Tensor:
    """ This error function calculates the difference between two Gaussian clouds, considering their mean position,
        scaling, rotation and color.
    """
    if len(optim_vars) == 1 and len(aux_vars) == 7:
        pose_j, = optim_vars
        pose_i, gaussian_xyz_i, gaussian_scaling_i, gaussian_rotation_i, gaussian_color_i, gaussian_xyz_j, gaussian_color_j = aux_vars
    elif len(optim_vars) == 2 and len(aux_vars) == 6:
        pose_i, pose_j = optim_vars
        gaussian_xyz_i, gaussian_scaling_i, gaussian_rotation_i, gaussian_color_i, gaussian_xyz_j, gaussian_color_j = aux_vars
    else:
        raise ValueError(f"Wrong input error function size, got {len(optim_vars)} and {len(aux_vars)}")

    scaling_factor = 1
    gaussian_xyz_i_corrected = gaussian_xyz_i.tensor @ pose_i.rotation().tensor.transpose(-1, -2) + pose_i.translation().tensor.unsqueeze(-2)
    gaussian_covariance_i = build_covariance_from_scaling_rotation(gaussian_scaling_i.tensor, scaling_factor, gaussian_rotation_i.tensor)
    gaussian_xyz_j_corrected = gaussian_xyz_j.tensor @ pose_j.rotation().tensor.transpose(-1, -2) + pose_j.translation().tensor.unsqueeze(-2)

    h_distance = hellinger_distance(gaussian_xyz_i_corrected, gaussian_covariance_i, gaussian_xyz_j_corrected, gaussian_covariance_i) # (batch_size, num_gs)
    color_diff = torch.norm(gaussian_color_i.tensor - gaussian_color_j.tensor, p=1, dim=-1) # (batch_size, num_gs)
    # valid_color = modified_sigmoid(color_diff, k=5) > 0.1
    return modified_sigmoid(color_diff, k=6) * h_distance.sqrt()


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size=0.05) -> o3d.geometry.PointCloud:
    """ Downsample the given point cloud, estimate normals, then compute a FPFH feature for each point
    Args:
        pcd: input point cloud
        voxel_size: size of voxels, inside which only one point will be sampled
    Return:
        pcd_down: down-sampled point cloud
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down


def compute_relative_pose(
        source_pcl: o3d.geometry.PointCloud,
        target_pcl: o3d.geometry.PointCloud,
        current_transformation: np.ndarray=None,
        voxel_size=0.05,
        distance_threshold=0.05
    ) -> np.ndarray:
    """ Caluculate the relative pose between two cameras using mean positions of Gaussian clouds
    inside the camera frustum, following a corse-to-fine process.
    """
    source_down = preprocess_point_cloud(source_pcl, voxel_size)
    target_down = preprocess_point_cloud(target_pcl, voxel_size)
    # refine the alignment with Point-to-plane ICP
    icp_pose = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return icp_pose.transformation, np.asarray(icp_pose.correspondence_set)
















########################################### ignore these functions ###############################################
def error_fn_dense_surface_alignment(
        optim_vars: Union[Tuple[th.SE3, th.SE3], Tuple[th.SE3, th.SE3, th.Variable]],
        aux_vars: Tuple[th.SE3, th.Point3]
    ) -> torch.Tensor:
        # determine whether the edge is odometry edge or loop closure edge
        tuple_size = len(optim_vars)
        if tuple_size == 2:
            pose_i, pose_j = optim_vars
        elif tuple_size == 3:
            pose_i, pose_j, l_ij = optim_vars
        else:
            raise ValueError(f"optim_vars tuple size is {tuple_size}, which can only be 2 or 3.")
        relative_pose, gaussian_xyz = aux_vars
        
        pose_i_hat = pose_j.compose(relative_pose) # (batch_size, 3, 4)
        rot_residual = pose_i.rotation().tensor - pose_i_hat.rotation().tensor # (batch_size, 3, 3)
        trans_residual = pose_i.translation().tensor - pose_i_hat.translation().tensor # (batch_size, 3)
        residual = rot_residual @ gaussian_xyz.tensor.transpose(1, 2) + trans_residual.unsqueeze(-1) # (batch_size, 3, data_size)
        res_norm = torch.norm(residual, p=2, dim=1, keepdim=False)
        if tuple_size == 3:
            return l_ij.tensor.sqrt() * res_norm
        else:
            return res_norm
    

def error_fn_dense_surface_alignment_v2(
        optim_vars: Union[Tuple[th.SE3, th.SE3], Tuple[th.SE3, th.SE3, torch.Tensor]],
        aux_vars: Tuple[th.SE3, th.Point3]
    ) -> torch.Tensor:
        """
        Compute the dense surface alignment error between two vertices, can be used as the error
        input to instantiate a th.CostFunction variable

        Args:
            optim_vars: optimizaiton variables registered in cost function, should contain
                vertex_i, vertex_j: correction matrix for pose i, j, 
                l_ij (optional): line process coefficient

            aux_vars: auxiliary variables registered in cost function, should contain
                relative_pose: constraint between vertex_i and vertex_j
                sgaussian_xyz: mean positions of the 3D Gaussians inside frustum i 
                    (gaussian means inside frustum j is not needed), shape=(num, 3)

        Returns:
            square root of global place recognition error
        """
        # determine whether the edge is odometry edge or loop closure edge
        tuple_size = len(optim_vars)
        if tuple_size == 2:
            pose_i, pose_j = optim_vars
        elif tuple_size == 3:
            pose_i, pose_j, l_ij = optim_vars
        else:
            raise ValueError(f"optim_vars tuple size is {tuple_size}, which can only be 2 or 3.")
        relative_pose, gaussian_xyz = aux_vars

        pose_ij_odometry : th.SE3 = pose_j.inverse().compose(pose_i) # (batch_size, 3, 4)
        pose_residual : th.SE3 = relative_pose.inverse().compose(pose_ij_odometry) # (batch_size, 3, 4)

        rot_residual = pose_residual.rotation().log_map().unsqueeze(1) # (batch_size, 1, 3)
        trans_residual = pose_residual.translation().tensor.unsqueeze(1) # (batch_size, 1, 3)
        xi = torch.cat((rot_residual, trans_residual), dim=-1) # (batch_size, 1, 6)

        p_skew_symmetric = to_skew_symmetric(gaussian_xyz.tensor) # (batch_size, num_pts, 3, 3)
        G_p = torch.cat(( # (batch_size, num, 3, 6)
            -p_skew_symmetric, 
            torch.tile(torch.eye(3), (gaussian_xyz.shape[0], gaussian_xyz.shape[-2], 1, 1)),
            ), dim=-1)
        Lambda = torch.sum(G_p.transpose(-2, -1) @ G_p, axis=1) # (batch_size, 6, 6)
        res = (xi @ Lambda @ xi.transpose(-2, -1)).squeeze(1) # (batch_size, 1)

        if tuple_size == 3:
            return l_ij.tensor.sqrt() * res.sqrt()
        else:
            return res.sqrt()


def to_skew_symmetric(tensor: torch.Tensor) -> torch.Tensor:
    """ Transform a (..., 3) tensor to a (..., 3, 3) skew-symmetric tensor
    Args:
        tensor: 3-vector(s) that need(s) to be transformed to skew symmetric matrix
    Returns:
        skew_symmetric: transformed skew symmetric matrices
    """
    size = tensor.size()
    if len(size) > 2 or size[-1] != 3:
        raise ValueError("Incorrect tensor dimension!")
    
    skew_symmetric = torch.zeros(tensor.shape+(3, ), dtype=tensor.dtype, device=tensor.device)

    skew_symmetric[..., 0, 1] = -tensor[..., 2]
    skew_symmetric[..., 0, 2] = tensor[..., 1]
    skew_symmetric[..., 1, 0] = tensor[..., 2]
    skew_symmetric[..., 1, 2] = -tensor[..., 0]
    skew_symmetric[..., 2, 0] = -tensor[..., 1]
    skew_symmetric[..., 2, 1] = tensor[..., 0]

    return skew_symmetric


def quaternion_multiplication(q1: torch.Tensor, q2: torch.Tensor):
    """ quaternion multiplication q1*q2 (q1 must be on left), broadcasting and batch operation supported """
    q1_w, q1_x, q1_y, q1_z = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    q2_w, q2_x, q2_y, q2_z = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = q1_w * q2_w - q1_x * q2_x - q1_y * q2_y - q1_z * q2_z
    x = q1_w * q2_x + q1_x * q2_w + q1_y * q2_z - q1_z * q2_y
    y = q1_w * q2_y - q1_x * q2_z + q1_y * q2_w + q1_z * q2_x
    z = q1_w * q2_z + q1_x * q2_y - q1_y * q2_x + q1_z * q2_w

    return torch.stack((w, x, y, z), dim=-1)


# th.OptimizerInfo.best_solution