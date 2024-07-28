import random
from typing import List, Union, Tuple
import numpy as np
import torch
import open3d as o3d
import theseus as th
from scipy.spatial import KDTree

from src.utils.utils import torch2np
from src.utils.gaussian_model_utils import build_scaling_rotation, strip_symmetric



def downsample(container, num_samples: int):
    """ downsample elements inside a container with a given number """
    num_before = len(container)
    if num_before <= num_samples:
        return container
    else:
        return random.sample(container, num_samples)


def match_gaussian_means(
    pts_1: torch.Tensor, 
    pts_2: torch.Tensor, 
    transformation: torch.Tensor,
    epsilon=5e-2
    ) -> List:
    """ Select inlier correspondences from two Gaussian clouds, use kd-tree to speed up """
    if transformation.size() != torch.Size([4, 4]):
        raise ValueError(f"The size of input transformation matrix must be (4, 4), but get {transformation.size()}")

    rotation = transformation[:3, :3]
    translation = transformation[:3, 3]
    pts_1_new = (rotation @ pts_1).squeeze() + translation

    pts_1_numpy = torch2np(pts_1_new)
    pts_2_numpy = torch2np(pts_2)
    pts2_kdtree = KDTree(pts_2_numpy)

    _, query_idx = pts2_kdtree.query(
        pts_1_numpy, 
        distance_upper_bound=epsilon, 
        workers=-1
    )
    res_list = []
    for i in range(query_idx.shape[0]):
        if query_idx[i] != pts_2.shape[0]:
            res_list.append(i)
    print(f'{len(res_list)} correspondences are found before adding to pose graph.')
    return res_list


def to_skew_symmetric(tensor: torch.Tensor):
    """
    Transform a (3, ) tensor to a (3, 3) tensor, or
    Transform a (n, 3) tensor to a (n, 3, 3) tensor, where n is the batch size

    Args:
        tensor (torch.Tensor): 3-vector(s) that need(s) to be transformed to skew symmetric matrix

    Returns:
        skew_symmetric (torch.Tensor): transformed skew symmetric matrices
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
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
        relative_pose, gaussian_means = aux_vars
        
        pose_i_hat = pose_j.compose(relative_pose) # (batch_size, 3, 4)
        rot_residual = pose_i.rotation().tensor - pose_i_hat.rotation().tensor # (batch_size, 3, 3)
        trans_residual = pose_i.translation().tensor - pose_i_hat.translation().tensor # (batch_size, 3)
        residual = rot_residual @ gaussian_means.tensor.transpose(1, 2) + trans_residual.unsqueeze(-1) # (batch_size, 3, data_size)
        res_norm = torch.norm(residual, p=2, dim=1, keepdim=False)
        if tuple_size == 3:
            return l_ij.tensor.sqrt() * res_norm
        else:
            return res_norm
    

# TODO: the gaussian_means must keep the same, do I need to detatch them and .to(cpu)?
def dense_surface_alignment(
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
                gaussian_means: mean positions of the 3D Gaussians inside frustum i 
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
        relative_pose, gaussian_means = aux_vars

        pose_ij_odometry : th.SE3 = pose_j.inverse().compose(pose_i) # (batch_size, 3, 4)
        pose_residual : th.SE3 = relative_pose.inverse().compose(pose_ij_odometry) # (batch_size, 3, 4)

        rot_residual = pose_residual.rotation().log_map().unsqueeze(1) # (batch_size, 1, 3)
        trans_residual = pose_residual.translation().tensor.unsqueeze(1) # (batch_size, 1, 3)
        xi = torch.cat((rot_residual, trans_residual), dim=-1) # (batch_size, 1, 6)

        p_skew_symmetric = to_skew_symmetric(gaussian_means.tensor) # (batch_size, num_pts, 3, 3)
        G_p = torch.cat(( # (batch_size, num, 3, 6)
            -p_skew_symmetric, 
            torch.tile(torch.eye(3), (gaussian_means.shape[0], gaussian_means.shape[-2], 1, 1)),
            ), dim=-1)
        Lambda = torch.sum(G_p.transpose(-2, -1) @ G_p, axis=1) # (batch_size, 6, 6)
        res = (xi @ Lambda @ xi.transpose(-2, -1)).squeeze(1) # (batch_size, 1)

        if tuple_size == 3:
            return l_ij.tensor.sqrt() * res.sqrt()
        else:
            return res.sqrt()
    

def error_fn_dense_gaussian_alignment(
    optim_vars: Union[Tuple[th.SE3, th.SE3], Tuple[th.SE3, th.SE3, torch.Tensor]],
    aux_vars: Tuple[th.SE3, th.Point3, th.Point3, th.Variable, th.Variable, th.Variable, th.Variable, th.Variable, th.Variable]
) -> torch.Tensor:
    """ """
    tuple_size = len(optim_vars)
    if tuple_size == 2:
        pose_i, pose_j = optim_vars
    elif tuple_size == 3:
        pose_i, pose_j, l_ij = optim_vars
    else:
        raise ValueError(f"optim_vars tuple size is {tuple_size}, which can only be 2 or 3.")
    
    relative_pose, gaussian_means_i, gaussian_scaling_i, gaussian_rotation_i, gaussian_color_i, \
        gaussian_means_j, gaussian_scaling_j, gaussian_rotation_j, gaussian_color_j = aux_vars
    
    # We want the corrected Gaussian cloud still has the same relative pose to another corrected Gaussian cloud
    gaussian_means_i_corrected = pose_i.transform_from(gaussian_means_i) # (batch_size, num_gs, 3)
    gaussian_rotation_i_corrected = pose_i.compose(gaussian_rotation_i) # （batch_size, num_gs, 3）TODO: debug
    gaussian_means_j_corrected = pose_j.transform_from(gaussian_means_j) # ()
    gaussian_rotation_j_corrected = pose_j.compose(gaussian_rotation_j)

    gaussian_means_i_transformed = relative_pose.transform_from(gaussian_means_i_corrected)
    gaussian_rotation_i_transformed = relative_pose.compose(gaussian_rotation_i_corrected)
    h_distance = hellinger_distance(
        gaussian_means_i_transformed.tensor, gaussian_scaling_i.tensor, gaussian_rotation_i_transformed.tensor,
        gaussian_means_j_corrected.tensor, gaussian_scaling_j.tensor, gaussian_rotation_j_corrected.tensor
    )
    color_diff = torch.norm(gaussian_color_i.tensor-gaussian_color_j.tensor, p=1)
    # TODO: do I need to square root the h_distance?
    if tuple_size == 2:
        return (color_diff * h_distance).squeeze()
    else:
        return (l_ij.tensor.sqrt() * color_diff * h_distance).squeeze()
    

def error_fn_line_process(optim_vars: th.Vector, aux_vars=None) -> torch.Tensor:
    """
    Computes the line process error of a loop closrue edge, can be used as the error
    input to instantiate a th.CostFunction variable
    Args:
        optim_vars:
            l_ij: jointly optimized weight (l_ij ∈ [0, 1]) over the loop edges
            (note that the scaling factor mu is considered as cost_weight)
    Returns:
        square root of line process error
    """
    l_ij, = optim_vars
    return l_ij.tensor.sqrt() - 1   


def point_cloud_registration(
    source_pcl: o3d.geometry.PointCloud,
    target_pcl: o3d.geometry.PointCloud,
    current_transformation=np.eye(4),
    voxel_size=0.05
    ) -> np.ndarray:
    # preprocess point cloud
    source_down = source_pcl.voxel_down_sample(voxel_size)
    target_down = target_pcl.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    distance_threshold = voxel_size * 0.4
    icp_pose = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return icp_pose.transformation


def hellinger_distance(
    gaussian_mean_i: torch.Tensor, gaussian_scaling_i: torch.Tensor, gaussian_rotation_i: torch.Tensor,
    gaussian_mean_j: torch.Tensor, gaussian_scaling_j: torch.Tensor, gaussian_rotation_j: torch.Tensor
):
    """ Computes the squared Hellinger distance between two gaussian distributions batch operation is supported"""
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        return L @ L.transpose(1, 2)
    
    device = gaussian_mean_i.device

    gaussian_scaling_activated_i = torch.exp(gaussian_scaling_i)
    gaussian_rotation_activated_i = torch.nn.functional.normalize(gaussian_rotation_i, dim=1)
    gaussian_covariance_i = build_covariance_from_scaling_rotation(gaussian_scaling_activated_i, 1, gaussian_rotation_activated_i).to(device)

    gaussian_scaling_activated_j = torch.exp(gaussian_scaling_j)
    gaussian_rotation_activated_j = torch.nn.functional.normalize(gaussian_rotation_j)
    gaussian_covariance_j = build_covariance_from_scaling_rotation(gaussian_scaling_activated_j, 1, gaussian_rotation_activated_j).to(device)

    gaussian_covariance_mean = (gaussian_covariance_i + gaussian_covariance_j) / 2
    det_gaussian_covariance_i = torch.linalg.det(gaussian_covariance_i)
    det_gaussian_covariance_j = torch.linalg.det(gaussian_covariance_j)
    det_gaussian_covariance_mean = torch.linalg.det(gaussian_covariance_mean)
    
    gaussian_mean_diff = gaussian_mean_i - gaussian_mean_j
    coefficient = det_gaussian_covariance_i.pow(1/4) * det_gaussian_covariance_j.pow(1/4) / det_gaussian_covariance_mean.pow(1/2)
    power = -1/8 * gaussian_mean_diff @ gaussian_covariance_mean.inverse() @ gaussian_mean_diff.transpose(0, 1)
    h_distance = 1 - coefficient * torch.exp(power)
    
    return h_distance.squeeze()


# th.OptimizerInfo.best_solution