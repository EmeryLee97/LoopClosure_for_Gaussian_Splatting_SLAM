import numpy as np
import torch
import open3d as o3d
import theseus as th
from typing import List, Union, Tuple, Optional

from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

from src.entities.gaussian_model import GaussianModel
from src.utils.mapper_utils import compute_camera_frustum_corners, compute_frustum_point_ids
from src.utils.utils import to_skew_symmetric, torch2np, np2torch, np2ptcloud, preprocess_point_cloud


def match_gaussian_means(
        pts_1: torch.tensor,
        pts_2: torch.tensor,
        transformation: torch.tensor,
        epsilon:float=5e-2
    ) -> List[Tuple[int, int]]:
    """
    Select inlier correspondences from two Gaussian clouds, use kd-tree to speed up

    Args:
        pts_1, pts_2: mean positions of 3D Gaussians
        transformation: prior transformation matrix from one Gaussian cloud to the other
        epsilon: threshold for finding inlier correspondence

    Returns:
        a list contains tuples of matching indices
    """
    if transformation.size() != torch.Size([4, 4]):
        raise ValueError(f"The size of input transformation matrix must be (4, 4), but get {transformation.size()}")

    if pts_1.size(-1) != 1:
        pts_1 = pts_1.unsqueeze(-1)

    rotation = transformation[:3, :3]
    translation = transformation[:3, 3]
    pts_1_new = (rotation @ pts_1).squeeze() + translation

    pts_1_numpy = torch2np(pts_1_new)
    pts_2_numpy = torch2np(pts_2)
    pts2_kdtree = KDTree(pts_2_numpy)

    _, query_idx = pts2_kdtree.query(pts_1_numpy, distance_upper_bound=epsilon, workers=-1)

    data_size = pts_1.size()[0]
    res_list = []
    for i in range(data_size):
        if query_idx[i] != data_size:
            res_list.append((i, query_idx[i]))

    return res_list


# TODO: select hyperparameters
# hyperparameters: voxel_size, distance_threshold_global, distance_threshold_local ...
def compute_relative_pose(
        depth_map_i: np.ndarray,
        pose_i: np.ndarray,
        gaussians_i: GaussianModel,
        depth_map_j: np.ndarray,
        pose_j: np.ndarray,
        gaussians_j: GaussianModel,
        intrinsics: np.ndarray,
        voxel_size: float
    ) -> np.ndarray:
    """ 
    https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html

    Caluculate the relative pose between two cameras using mean positions of Gaussian clouds
    inside the camera frustum, following a corse-to-fine process.

    Args:
        depth_map_i, depth_map_j: ground truth depth map of the current keyframe, used to 
            determine the near an far plane of the frustum
        pose_i, pose_j: camera pose of the current keyframe
        gaussians_i, gaussians_j: gaussian clouds for registration
        intrinsics: camera intrinsic matrix of the dataset
        voxel_size: voxel size for point cloud down sampling

    Returns:
        relative pose between two keyframes
    """
    gaussian_points_i = gaussians_i.get_xyz()
    camera_frustum_corners_i = compute_camera_frustum_corners(depth_map_i, pose_i, intrinsics)
    reused_pts_ids_i = compute_frustum_point_ids(
            gaussian_points_i, np2torch(camera_frustum_corners_i), device=gaussian_points_i.device)
    point_cloud_i = np2ptcloud(gaussian_points_i[reused_pts_ids_i])
    pcd_down_i, pcd_fpfh_i = preprocess_point_cloud(point_cloud_i, voxel_size)

    gaussian_points_j = gaussians_j.get_xyz()
    camera_frustum_corners_j = compute_camera_frustum_corners(depth_map_j, pose_j, intrinsics)
    reused_pts_ids_j = compute_frustum_point_ids(
            gaussian_points_j, np2torch(camera_frustum_corners_j), device=gaussian_points_j.device)
    point_cloud_j = np2ptcloud(gaussian_points_j[reused_pts_ids_j])
    pcd_down_j, pcd_fpfh_j = preprocess_point_cloud(point_cloud_j, voxel_size)

    # use RANSAC for global registration on a heavily down-sampled point cloud
    distance_threshold_global = voxel_size * 1.5
    print(f"RANSAC registration on downsampled point clouds with a liberal distance threshold {distance_threshold_global}")
    ransac_pose = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_down_i, pcd_down_j, pcd_fpfh_i, pcd_fpfh_j, distance_threshold_global,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_global)], 
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
    # use Point-to-plane ICP to further refine the alignment
    distance_threshold_local = voxel_size * 0.4
    print(f"Point-to-plane ICP registration is applied on original point clouds to refine the alignment \
           with a strict distance threshold {distance_threshold_local}")
    icp_pose = o3d.pipelines.registration.registration_icp(
        point_cloud_i, point_cloud_j, distance_threshold_local, ransac_pose.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return icp_pose.transformation


# TODO: the gaussian_means must keep the same, do I need to detatch them and .to(cpu)?
#       How to make sure l_ij is between [0, 1]? test this function
def dense_surface_alignment_old(
        optim_vars: Union[Tuple[th.SE3, th.SE3], Tuple[th.SE3, th.SE3, torch.Tensor]],
        aux_vars: Tuple[th.Variable, th.Variable]
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
                    (gaussian means inside frustum j is not needed)

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
        relative_pose_th, gaussian_means_th = aux_vars
        relative_pose = relative_pose_th.tensor
        gaussian_means = gaussian_means_th.tensor

        if pose_i.shape[0] != 1 or pose_i.shape[0] != 1:
            raise ValueError(f"Expected batch size = 1, but got {pose_i.shape[0]}.")
        
        pose_mat: torch.Tensor = torch.inverse(relative_pose) @ torch.inverse(pose_j) @ pose_i
        rot_mat: np.ndarray = torch2np(pose_mat[:3, :3])
        rot:Rotation = Rotation.from_matrix(rot_mat)
        axis_angle_rotation: np.ndarray = rot.as_rotvec()
        axis_angle: torch.Tensor = torch.Tensor(axis_angle_rotation)
        trans: torch.Tensor = pose_mat[:3, 3]

        xi = torch.cat((axis_angle, trans))

        p_skew_symmetric = to_skew_symmetric(gaussian_means) # (n, 3， 3) tensor
        G_p = torch.cat((-p_skew_symmetric, torch.eye(3).unsqueeze(0).expand(gaussian_means.size()[0], -1, -1)), dim=-1) # (n, 3, 6) tensor
        G_p_square = torch.matmul(G_p.transpose(1, 2), G_p) # (n, 6, 6) tensor
        Lambda = torch.sum(G_p_square, dim=0) # (6, 6) tensor
        res = xi @ Lambda @ xi

        if tuple_size == 3:
            return l_ij.sqrt() * res.sqrt()
        else:
            return res.sqrt()


def dense_surface_alignment(
        optim_vars: Union[Tuple[th.SE3, th.SE3], Tuple[th.SE3, th.SE3, th.Vector]],
        aux_vars: Tuple[th.SE3, th.Point3]
    ) -> torch.Tensor:
    """
    Compute the dense surface alignment error between two vertices, can be used as the error
    function input to instantiate a th.CostFunction variable

    Args:
        optim_vars: optimizaiton variables registered in cost function, should contain
            pose_i, pose_j: correction matrix for pose i, j
            l_ij (optional): line process coefficient

        aux_vars: auxiliary variables registered in cost function, should contain
            relative_pose: constraint between vertex_i and vertex_j
            gaussian_means_i: mean positions of the 3D Gaussians inside camera frustum, 
                represented in coordinate i and coordinate (those in coordinate j are not needed)

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
    relative_pose, gaussian_means_i = aux_vars

    # transform all points in coordinate i and j to world coordinate
    gaussian_means_i_transformed: th.Point3 = pose_i.transform_from(gaussian_means_i)
    gaussian_means_j_transformed: th.Point3 = pose_j.transform_from(
        relative_pose.transform_from(gaussian_means_i))

    residual = (gaussian_means_i_transformed - gaussian_means_j_transformed).tensor

    # check if this error function is used for odometry edge or loop edge
    if tuple_size == 2:
        return residual
    else:
        return torch.sqrt(l_ij.tensor) * residual


def line_process(optim_vars: th.Vector, aux_vars=None) -> torch.Tensor:
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


"""
    Custom Cost Funtion inherits from 'th.CostFunction'
    contains 
        optimization variables inherits from 'th.Manifold'
        auxiliary variables
    implements
        error computation and Jacobian
    returns
        torch.tensor as its error
"""
# TODO: implement the jacobians() method
class AdjacentVerticeCost(th.CostFunction):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        vertex_i: th.SE3,
        vertex_j: th.SE3,
        relative_pose = torch.eye(4),
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)

        if not isinstance(vertex_i, th.SE3) or not isinstance(vertex_j, th.SE3):
            raise TypeError("Vertex should be a th.SE3.")
        if not vertex_i.dof() == vertex_j.dof():
            raise ValueError("Two vertices must have identical dof.")

        self._cost_weight = cost_weight
        self._vertex_i = vertex_i
        self._vertex_j = vertex_j
        self._relative_pose = relative_pose
        self.register_optim_vars(["vertex_i", "vertex_j"])
        # self.register_aux_vars([ ])

    def error(self, gaussian_means: torch.tensor) -> torch.Tensor:
        """
        Compute the edge constraint between two adjacent vertices
        Returns:
            global place recognition error
        """
        return dense_surface_alignment(self._vertex_i, self._vertex_j, self._relative_pose, gaussian_means)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # TODO: implementation
        """
        Returns:
            jacobians: a list of jacobians, The i-th jacobian has shape 
                       (batch_size, cf.dim(), i-th_optim_var.dof())
            error: self.error()
        """
        pass

    def dim(self) -> int:
        return self._vertex_i.dof

    def _copy_impl(self, new_name: Optional[str] = None) -> "AdjacentVerticeCost":
        return AdjacentVerticeCost(
            self._cost_weight.copy(), self._vertex_i.copy(), self._vertex_j.copy(), name=new_name
            )

# not implemented
class LoopClosureVerticeCost(th.CostFunction):
    def __init__():
        pass

    def error(self, gaussian_means: torch.tensor) -> torch.Tensor:
        pass

    def dim(self) -> int:
        return self._vertex_i.dof + 1
    
    def _copy_impl(self):
        pass


class GaussianSLAMEdge:
    def __init__(
        self,
        vertex_idx_i: int,
        vertex_idx_j: int,
        relative_pose: th.SE3,
        cost_weight: th.CostWeight
    ):
        self.vertex_idx_i = vertex_idx_i
        self.vertex_idx_j = vertex_idx_j
        self.relative_pose = relative_pose
        self.cost_weight = cost_weight

    def to(self, *args, **kwargs):
        self.weight.to(*args, **kwargs)
        self.relative_pose.to(*args, **kwargs)


class GaussianSLAMPoseGraph:
    def __init__(
        self, 
        requires_auto_grad = True
    ):
        self._requires_auto_grad = requires_auto_grad
        self._objective = th.Objective()
        self._theseus_inputs = {} 

    def add_odometry_edge(
            self,
            vertex_i: th.SE3,
            vertex_j: th.SE3,
            edge: GaussianSLAMEdge,
            gaussian_means: torch.Tensor
        ):

        relative_pose = edge.relative_pose
        cost_weight = edge.cost_weight

        #gaussian_means_th = th.Point3(
            #tensor=gaussian_means, 
            #name=f"gaussian_means_odometry__{edge.vertex_idx_i}_{edge.vertex_idx_j}")
        optim_vars = vertex_i, vertex_j
        #aux_vars = relative_pose, gaussian_means_th

        if self._requires_auto_grad:
            for point in gaussian_means:
                point_th = th.Point3(tensor=point)
                aux_vars = relative_pose, point_th
                cost_function = th.AutoDiffCostFunction(
                    optim_vars, dense_surface_alignment, 3, cost_weight, aux_vars
                )
                self._objective.add(cost_function)

            self._theseus_inputs.update({
                vertex_i.name: vertex_i.tensor, 
                vertex_j.name: vertex_j.tensor,
            })
        else:
            raise NotImplementedError()

    def add_loop_closure_edge(
            self,
            vertex_i: th.SE3,
            vertex_j: th.SE3,
            edge: GaussianSLAMEdge,
            gaussian_means: torch.tensor,
            coefficient: float # hyperparameter, not the same as in the paper
        ):

        relative_pose = edge.relative_pose
        cost_weight_alignment = edge.cost_weight

        l_ij = th.Vector(
            tensor=torch.ones(1, 1), 
            name=f"line_process__{edge.vertex_idx_i}_{edge.vertex_idx_j}"
        )
        #gaussian_means_th = th.Point3(
            #tensor=gaussian_means, 
            #name=f"gaussian_means_odometry__{edge.vertex_idx_i}_{edge.vertex_idx_j}")

        optim_vars = vertex_i, vertex_j, l_ij
        #aux_vars = relative_pose, gaussian_means_th

        cost_weight_line_process = th.ScaleCostWeight(coefficient)

        if self._requires_auto_grad:
            for point in gaussian_means:
                point_th = th.Point3(tensor=point)
                aux_vars = relative_pose, point_th
                cost_function = th.AutoDiffCostFunction(
                    optim_vars, dense_surface_alignment, 3, cost_weight_alignment, aux_vars
                )
                self._objective.add(cost_function)
                
            # auxiliary variables can be not declared
            optim_vars = l_ij,
            cost_function = th.AutoDiffCostFunction(
                optim_vars, line_process, 1, cost_weight_line_process
            )
            self._objective.add(cost_function)
        
            self._theseus_inputs.update({
                vertex_i.name: vertex_i.tensor, 
                vertex_j.name: vertex_j.tensor,
                l_ij.name: l_ij.tensor,
            })
        else:
            raise NotImplementedError()
        
    def optimize(self, max_iterations=1e3, step_size=0.01, track_best_solution=True, verbose=False):
        optimizer = th.LevenbergMarquardt(
            objective=self._objective,
            max_iterations=max_iterations,
            step_size=step_size)
        
        layer = th.TheseusLayer(optimizer)

        with torch.no_grad():
            _, info = layer.forward(
                self._theseus_inputs, 
                optimizer_kwargs={"track_best_solution":track_best_solution, "verbose":verbose}
                )
        return info

    def to(self, *args, **kwargs):
        if self._poses is not None:
            for pose in self.poses:
                pose.to(*args, **kwargs)

        if self._edges is not None:
            for edge in self.edges:
                edge.to(*args, **kwargs)


