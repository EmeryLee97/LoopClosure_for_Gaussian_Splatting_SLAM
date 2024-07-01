import numpy as np
import torch
import open3d as o3d
import theseus as th
from typing import List, Union, Tuple, Dict

#from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

from src.entities.gaussian_model import GaussianModel
from src.utils.mapper_utils import compute_camera_frustum_corners, compute_frustum_point_ids
from src.utils.utils import to_skew_symmetric, torch2np, np2torch, np2ptcloud, preprocess_point_cloud


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


class GaussianSLAMEdge:
    def __init__(
        self,
        vertex_idx_i: int,
        vertex_idx_j: int,
        relative_pose: torch.Tensor,
        cost_weight=1.0
    ):
        self.vertex_idx_i = vertex_idx_i
        self.vertex_idx_j = vertex_idx_j
        self.relative_pose = relative_pose
        self.cost_weight = th.ScaleCostWeight(
            scale=cost_weight,
            name=f"EDGE_WEIGHT__{str(vertex_idx_i).zfill(6)}_{str(vertex_idx_j).zfill(6)}"
        )


class GaussianSLAMPoseGraph:
    def __init__(
        self, 
        config: Dict, 
        requires_auto_grad = True
    ):
        self._requires_auto_grad = requires_auto_grad
        self.objective = th.Objective()

        self.center_matching_threshold = config["center_matching_threshold"]
        self.optimization_max_iterations = config["optimization_max_iterations"]
        self.optimization_step_size = config["optimization_step_size"]
        self.loop_inlier_threshold = config["loop_inlier_threshold"]
        self.correspondence_factor = config["correspondence_factor"]
        self.damping = config["damping"]
        self.track_best_solution = config["track_best_solution"]
        self.verbose = config["verbose"]

    def add_odometry_edge(
            self,
            vertex_i: th.SE3,
            vertex_j: th.SE3,
            edge: GaussianSLAMEdge,
            gaussian_means: torch.Tensor
        ):
        """ add an odometry constraint to the objective"""
        gaussian_means_th = th.Variable(
            tensor=gaussian_means.unsqueeze(0), 
            name=f"gaussian_means_odometry__{str(edge.vertex_idx_i).zfill(6)}_{str(edge.vertex_idx_j).zfill(6)}"
        )

        if self._requires_auto_grad:
            cost_fn = th.AutoDiffCostFunction(
                optim_vars=[vertex_i, vertex_j],
                err_fn=GaussianSLAMPoseGraph.dense_surface_alignment, 
                dim=1, 
                cost_weight=edge.cost_weight, 
                aux_vars=[edge.relative_pose, gaussian_means_th],
                name=f"COST_FUNCTION_REGISTRATION__{str(edge.vertex_idx_i).zfill(6)}_{str(edge.vertex_idx_j).zfill(6)}"
            )
            self.objective.add(cost_fn)
        else:
            raise NotImplementedError()

    def add_loop_closure_edge(
            self,
            vertex_i: th.SE3,
            vertex_j: th.SE3,
            edge: GaussianSLAMEdge,
            gaussian_means: torch.Tensor, # inlier matches
        ) -> None:
        """ Add a loop constraint to the objective
        Args:
            vertex_i: absolute pose of a loop frame
            vertex_j: absolute pose of the current frame
            edge: loop edge with measured relative pose betwwen those frames
            gaussian_means: selected inlier machings, can be from either submap
            tau: fairly liberal distance threshold
        """
        num_matches = gaussian_means.shape[0]
        cost_weight_registration = edge.cost_weight # for dense surface alignment
        cost_weight_mu = cost_weight_registration.scale.tensor.squeeze() * np.sqrt(num_matches) * self.correspondence_factor
        cost_weight_line_process = th.ScaleCostWeight(cost_weight_mu) # for line process

        l_ij = th.Vector(
            tensor=torch.ones(1, 1), 
            name=f"LINE_PROCESS__{str(edge.vertex_idx_i).zfill(6)}_{str(edge.vertex_idx_j).zfill(6)}"
        )
        gaussian_means_th = th.Variable(
            tensor=gaussian_means.unsqueeze(0), 
            name=f"gaussian_means_odometry__{str(edge.vertex_idx_i).zfill(6)}_{str(edge.vertex_idx_j).zfill(6)}")

        if self._requires_auto_grad:
            cost_fn_registration = th.AutoDiffCostFunction(
                optim_vars=[vertex_i, vertex_j, l_ij], 
                err_fn=GaussianSLAMPoseGraph.dense_surface_alignment, 
                dim=1, 
                cost_weight=cost_weight_registration, 
                aux_vars=[edge.relative_pose, gaussian_means_th],
                name=f"COST_FUNCTION_LINE_PROCRSS__{str(edge.vertex_idx_i).zfill(6)}_{str(edge.vertex_idx_j).zfill(6)}"
            )
            self.objective.add(cost_fn_registration)

            cost_fn_line_process = th.AutoDiffCostFunction(
                optim_vars=[l_ij,], 
                err_fn=GaussianSLAMPoseGraph.line_process, 
                dim=1, 
                cost_weight=cost_weight_line_process,
                name=f"COST_FUNCTION_LINE_PROCRSS__{str(edge.vertex_idx_i).zfill(6)}_{str(edge.vertex_idx_j).zfill(6)}"
            )
            self.objective.add(cost_fn_line_process)
        else:
            raise NotImplementedError()

    def _remove_loop_outlier(self, substring="LINE_PROCESS") -> bool:
        """ Remove from objective false loops and all cost functions connect to them """
        flag = False
        for optim_key, optim_val in self.objective.optim_vars.items():
            if substring in optim_key:
                if optim_val.tensor < self.loop_inlier_threshold:
                    flag = True
                    print(f"Removing optimizaiton variable {optim_key} from objective.")
                    cost_fns = self.objective.get_functions_connected_to_optim_var(optim_val)
                    for cost_fn in cost_fns.copy():
                        print(f"Removing cost function {cost_fn.name} from objective.")
                        self.objective.erase(cost_fn.name)
        return flag
                    
    def _optimize(self) -> th.OptimizerInfo:
        optimizer = th.LevenbergMarquardt(
            objective=self.objective,
            max_iterations=self.optimization_max_iterations,
            step_size=self.optimization_step_size,
            linearization_cls=th.SparseLinearization,
            linear_solver_cls=th.CholmodSparseSolver,
            vectorize=True,
        )
        layer = th.TheseusLayer(optimizer)
        with torch.no_grad():
            _, info = layer.forward(
                optimizer_kwargs={
                    "damping": self.damping, 
                    "track_best_solution": self.track_best_solution, 
                    "verbose": self.verbose
                }
            )
        return info

    def optimize_two_steps(self) -> th.OptimizerInfo:
        """ optimization in two steps: 
        1. optimize with initial guess of all optim variables (T_i, l_ij)
        2. remove all l_ij < threshold, and all cost functions that are connected to them,
           optimize again with optimized variables
        """
        print(f"First step optimization, dealing with {self.objective.size_cost_functions()} cost functions")
        info = self._optimize()
        has_loop_outlier = self._remove_loop_outlier("LINE_PROCESS")
        if has_loop_outlier:
            print(f"Second step optimization, dealing with {self.objective.size_cost_functions()} cost functions")
            info = self._optimize()
        return info

    def match_gaussian_means(
            self,
            pts_1: torch.Tensor,
            pts_2: torch.Tensor,
            transformation: torch.Tensor,
        ) -> List[Tuple[int, int]]:
        """ Select inlier correspondences from two Gaussian clouds, use kd-tree to speed up
        Args:
            pts_1, pts_2: mean positions of 3D Gaussians
            transformation: prior transformation matrix from one Gaussian cloud to the other
        Return:
            a list contains tuples of matching indices, and the number of matching points
        """
        if transformation.size() != torch.Size([4, 4]):
            raise ValueError(f"The size of input transformation matrix must be (4, 4), but get {transformation.size()}")

        if pts_1.size(-1) != 1:
            pts_1 = pts_1.unsqueeze(-1)

        if isinstance(pts_1, th.Point3) or isinstance(pts_2, th.Point3):
            raise TypeError("To be matched points must be torch.Tensor")

        rotation = transformation[:3, :3]
        translation = transformation[:3, 3]
        pts_1_new = (rotation @ pts_1).squeeze() + translation

        pts_1_numpy = torch2np(pts_1_new)
        pts_2_numpy = torch2np(pts_2)
        pts2_kdtree = KDTree(pts_2_numpy)

        _, query_idx = pts2_kdtree.query(
            pts_1_numpy, 
            distance_upper_bound=self.center_matching_threshold, 
            workers=-1
        )

        data_size = pts_1.size()[0]
        print(f"{data_size} matching points are detected!")
        res_list = []
        for i in range(data_size):
            if query_idx[i] != data_size:
                res_list.append((i, query_idx[i]))

        return res_list

    # TODO: the gaussian_means must keep the same, do I need to detatch them and .to(cpu)?
    #       How to make sure l_ij is between [0, 1]? test this function
    @ staticmethod
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
        
    @ staticmethod
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
    

    
# th.OptimizerInfo.best_solution