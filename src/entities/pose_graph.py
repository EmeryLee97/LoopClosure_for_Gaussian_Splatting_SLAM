""" This module includes the GaussianSLAMEdge class and the GaussianSLAMPoseGraph class """

from typing import List, Dict, Type

import numpy as np
import torch
import theseus as th

from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.datasets import BaseDataset
from src.utils.gaussian_model_utils import SH2RGB

from src.utils.utils import np2torch, torch2np, np2ptcloud
from src.utils.mapper_utils import compute_camera_frustum_corners, compute_frustum_point_ids
from src.utils.pose_graph_utils import error_fn_dense_gaussian_alignment, match_gaussian_means, \
                                        downsample, compute_relative_pose


class GaussianSLAMEdge:
    def __init__(
        self, 
        vertex_idx_i: int, 
        vertex_idx_j: int, 
        relative_pose: torch.Tensor, 
        cost_weight=1.0, 
        device='cuda:0'
    ):
        # be careful of data type (torch.float64 and torch.float32)
        self.device = device
        self.vertex_idx_i = vertex_idx_i
        self.vertex_idx_j = vertex_idx_j
        self.relative_pose = relative_pose
        self.relative_pose.to(self.device)
        self.cost_weight = th.ScaleCostWeight(
            scale=cost_weight,
            name=f"EDGE_WEIGHT__{str(vertex_idx_i).zfill(6)}_{str(vertex_idx_j).zfill(6)}"
        )
        self.cost_weight.to(self.device)


class GaussianSLAMPoseGraph:
    def __init__(
        self, 
        config: Dict, 
        dataset: Type[BaseDataset], 
        logger: Logger,
        device='cuda:0', 
        requires_auto_grad=True
    ):
        self.dataset = dataset
        self.logger = logger
        self._requires_auto_grad = requires_auto_grad
        self.device = device
        self.objective = th.Objective()
        self.objective.to(self.device)

        self.use_gt_relative_pose = config["gt_relative_pose"]
        # self.opacity_threshold = config["opacity_threshold"]
        self.odo_matching_threshold = config["odo_matching_threshold"]
        self.loop_matching_threshold = config["loop_matching_threshold"]
        self.downsample_num = config["downsample_number"]
        self.odometry_weight = config["odometry_weight"]
        self.loop_weight = config["loop_weight"]
        self.optimization_max_iterations = config["optimization_max_iterations"]
        self.optimization_step_size = config["optimization_step_size"]
        self.rel_err_tolerance = config["rel_err_tolerance"]
        self.damping = config["damping"]
        self.track_best_solution = config["track_best_solution"]
        self.verbose = config["verbose"]

    def add_edge(
            self, 
            vertex_i: th.SE3, 
            vertex_j: th.SE3, 
            edge: GaussianSLAMEdge,
            gaussian_xyz_i: torch.Tensor, 
            gaussian_scaling_i: torch.Tensor, 
            gaussian_rotation_i: torch.Tensor, 
            gaussian_color_i: torch.Tensor,
            gaussian_color_j: torch.Tensor,
        ) -> None:
        """ add an odometry edge or loop edge to the objective """
        edge.relative_pose = edge.relative_pose.to(gaussian_xyz_i.device)
        gaussian_xyz_j = gaussian_xyz_i @ edge.relative_pose[:3, :3].transpose(-1, -2) + edge.relative_pose[:3, 3].unsqueeze(-2)
        num_matches = gaussian_xyz_i.shape[0]

        gaussian_xyz_i_th = th.Variable(tensor=gaussian_xyz_i.unsqueeze(0)) # (1, num_gs, 3)
        gaussian_scaling_i_th = th.Variable(tensor=gaussian_scaling_i.unsqueeze(0)) # (1, num_gs, 3)
        gaussian_rotation_i_th = th.Variable(tensor=gaussian_rotation_i.unsqueeze(0)) # (1, num_gs, 4)
        gaussian_color_i_th = th.Variable(tensor=gaussian_color_i.squeeze().unsqueeze(0)) # (1, num_gs, 3)
        gaussian_xyz_j_th = th.Variable(tensor=gaussian_xyz_j.unsqueeze(0)) # (1, num_gs, 3)
        gaussian_color_j_th = th.Variable(tensor=gaussian_color_j.squeeze().unsqueeze(0)) # (1, num_gs, 3)

        gaussian_xyz_i_th.to(self.device)
        gaussian_scaling_i_th.to(self.device)
        gaussian_rotation_i_th.to(self.device)
        gaussian_color_i_th.to(self.device)
        gaussian_xyz_j_th.to(self.device)
        gaussian_color_j_th.to(self.device)
        vertex_i.to(self.device)
        vertex_j.to(self.device)

        # vertex_0 should not be optimzied
        if edge.vertex_idx_i == 0:
            optim_vars = [vertex_j, ]
            aux_vars = [vertex_i, gaussian_xyz_i_th, gaussian_scaling_i_th, gaussian_rotation_i_th, 
                        gaussian_color_i_th, gaussian_xyz_j_th, gaussian_color_j_th]
        else:
            optim_vars = [vertex_i, vertex_j]
            aux_vars = [gaussian_xyz_i_th, gaussian_scaling_i_th, gaussian_rotation_i_th, 
                        gaussian_color_i_th, gaussian_xyz_j_th, gaussian_color_j_th]

        if self._requires_auto_grad:
            cost_fn = th.AutoDiffCostFunction(
                optim_vars=optim_vars,
                err_fn=error_fn_dense_gaussian_alignment, 
                dim=num_matches,
                cost_weight=edge.cost_weight, 
                aux_vars=aux_vars,
                name=f"COST_FUNCTION__{str(edge.vertex_idx_i).zfill(6)}_{str(edge.vertex_idx_j).zfill(6)}"
            )
            self.objective.add(cost_fn)
        else:
            raise NotImplementedError()

    def optimize(self) -> th.OptimizerInfo:
        optimizer = th.LevenbergMarquardt(
            objective=self.objective,
            max_iterations=self.optimization_max_iterations,
            step_size=self.optimization_step_size,
            linearization_cls=th.SparseLinearization,
            linear_solver_cls=th.CholmodSparseSolver,
            rel_err_tolerance=self.rel_err_tolerance,
            vectorize=True,
        )
        layer = th.TheseusLayer(optimizer)
        layer.to(self.device)
        with torch.no_grad():
            print(f"Optimization, dealing with {self.objective.size_cost_functions()} cost functions")
            _, info = layer.forward(optimizer_kwargs={
                "damping": self.damping, 
                "track_best_solution": self.track_best_solution, 
                "verbose": self.verbose
            })
        return info

    def create_odometry_constraint(
            self, 
            current_gaussian_model: GaussianModel, 
            last_gaussian_model: GaussianModel,
            current_submap_id: int,
            new_submap_frame_ids: List,
            estimated_c2ws: List, 
        ) -> None:
        """ create an odometry constraint between the last submap and the current submap, and add it into pose graph.
        Each vertex is initialized as identity matrix, with name 'VERTEX_SE3__str(i).zfill(6)', where 'i' indicates it's 
        the i'th submap. The relative pose between adjacent submaps is also initialized to identity matrix. 
        """
        last_submap_id = current_submap_id - 1
        last_submap_frame_id = new_submap_frame_ids[last_submap_id]
        print(f"First frame idx of submap_{last_submap_id} is {last_submap_frame_id}")
        last_frustum_corners = compute_camera_frustum_corners(self.dataset[last_submap_frame_id][2], estimated_c2ws[last_submap_frame_id], self.dataset.intrinsics)
        last_reused_pts_ids = compute_frustum_point_ids(last_gaussian_model.get_xyz(), last_frustum_corners, device=self.device)
        if self.objective.has_optim_var(f"VERTEX_SE3__{str(last_submap_id).zfill(6)}"):
            last_vertex = self.objective.get_optim_var(f"VERTEX_SE3__{str(last_submap_id).zfill(6)}")
        else:
            last_vertex = th.SE3(tensor=torch.tile(torch.eye(3, 4), [1, 1, 1]), name=f"VERTEX_SE3__{str(last_submap_id).zfill(6)}")

        current_submap_frame_id = new_submap_frame_ids[current_submap_id]
        print(f"First frame idx of submap_{current_submap_id} is {current_submap_frame_id}")
        current_frustum_corners = compute_camera_frustum_corners(self.dataset[current_submap_frame_id][2], estimated_c2ws[current_submap_frame_id], self.dataset.intrinsics)
        current_reused_pts_ids = compute_frustum_point_ids(current_gaussian_model.get_xyz(), current_frustum_corners, device=self.device)
        if self.objective.has_optim_var(f"VERTEX_SE3__{str(current_submap_id).zfill(6)}"):
            current_vertex = self.objective.get_optim_var(f"VERTEX_SE3__{str(current_submap_id).zfill(6)}")
        else:
            current_vertex = th.SE3(tensor=torch.tile(torch.eye(3, 4), [1, 1, 1]), name=f"VERTEX_SE3__{str(current_submap_id).zfill(6)}")

        print(f"Creating_odometry_constraint between submap_{last_submap_id} and submap_{current_submap_id}")
        match_idx_last, match_idx_current = match_gaussian_means(
            last_gaussian_model.get_xyz()[last_reused_pts_ids], 
            current_gaussian_model.get_xyz()[current_reused_pts_ids], 
            torch.eye(4, device="cuda"), 
            self.odo_matching_threshold
        )
        if len(match_idx_last) == 0:
            raise ValueError(f"No Gaussian correspondences found, please increase the match threshold or debug!")
        downsample_ids = downsample(match_idx_last, self.downsample_num*2)
        odometry_edge = GaussianSLAMEdge(last_submap_id, current_submap_id, torch.eye(3, 4), self.odometry_weight)
        print(f"Building odometry constraint between submap_{last_submap_id} and submap_{current_submap_id}")
        self.add_edge(
            last_vertex, current_vertex, odometry_edge, 
            last_gaussian_model.get_xyz()[last_reused_pts_ids][match_idx_last][downsample_ids], 
            last_gaussian_model.get_scaling()[last_reused_pts_ids][match_idx_last][downsample_ids], 
            last_gaussian_model.get_rotation()[last_reused_pts_ids][match_idx_last][downsample_ids], 
            SH2RGB(last_gaussian_model.get_features()[last_reused_pts_ids][match_idx_last][downsample_ids]).clamp(0, 1),
            SH2RGB(current_gaussian_model.get_features()[current_reused_pts_ids][match_idx_current][downsample_ids]).clamp(0, 1),
        )
        print(f"Current error metric = {self.objective.error_metric()}")

    def create_loop_constraint(
        self, 
        current_gaussian_model: GaussianModel, 
        loop_gaussian_model: GaussianModel,
        loop_submap_id: int, 
        current_submap_id: int,
        new_submap_frame_ids: List, 
        estimated_c2ws: torch.Tensor,
    ) -> None:
        """ create a loop constraint between the current submap and a submap that forms a loop with the current one, then
        add it into pose graph. Each vertex is initialized as identity matrix, with name 'VERTEX_SE3__str(i).zfill(6)', 
        where 'i' indicates it's the i'th submap.
        """
        print(f"There are {loop_gaussian_model.get_xyz().shape[0]} Gaussians in the loop submap")
        current_submap_frame_id = new_submap_frame_ids[current_submap_id]
        current_frustum_corners = compute_camera_frustum_corners(self.dataset[current_submap_frame_id][2], estimated_c2ws[current_submap_frame_id], self.dataset.intrinsics)
        current_reused_pts_ids = compute_frustum_point_ids(current_gaussian_model.get_xyz(), current_frustum_corners, device=self.device)
        if self.objective.has_optim_var(f"VERTEX_SE3__{str(current_submap_id).zfill(6)}"):
            current_vertex = self.objective.get_optim_var(f"VERTEX_SE3__{str(current_submap_id).zfill(6)}")
        else:
            current_vertex = th.SE3(tensor=torch.tile(torch.eye(3, 4), [1, 1, 1]), name=f"VERTEX_SE3__{str(current_submap_id).zfill(6)}")
        
        loop_submap_frame_id = new_submap_frame_ids[loop_submap_id]
        loop_frustum_corners = compute_camera_frustum_corners(self.dataset[loop_submap_frame_id][2], estimated_c2ws[loop_submap_frame_id], self.dataset.intrinsics)
        loop_reused_pts_ids = compute_frustum_point_ids(loop_gaussian_model.get_xyz(), loop_frustum_corners, device=self.device)
        if loop_submap_id == 0:
            loop_vertex = self.objective.get_aux_var(f"VERTEX_SE3__{str(loop_submap_id).zfill(6)}")
        else:
            loop_vertex = self.objective.get_optim_var(f"VERTEX_SE3__{str(loop_submap_id).zfill(6)}")
            
        if self.use_gt_relative_pose:
            current_correction_pose_gt = np2torch(self.dataset[current_submap_frame_id][-1]) @ estimated_c2ws[current_submap_frame_id].inverse()
            loop_correction_pose_gt = np2torch(self.dataset[loop_submap_frame_id][-1]) @ estimated_c2ws[loop_submap_frame_id].inverse()
            relative_pose = current_correction_pose_gt @ loop_correction_pose_gt.inverse()
            match_idx_loop, match_idx_current = match_gaussian_means(
            loop_gaussian_model.get_xyz()[loop_reused_pts_ids], # TODO: should I detach them from the graph?
            current_gaussian_model.get_xyz()[current_reused_pts_ids], 
            relative_pose.to('cuda'), 
            self.loop_matching_threshold
        )
        else:
            relative_pose, corres_set = compute_relative_pose(
                np2ptcloud(torch2np(loop_gaussian_model.get_xyz()[loop_reused_pts_ids])),
                np2ptcloud(torch2np(current_gaussian_model.get_xyz()[current_reused_pts_ids])),
                np.eye(4), 
                voxel_size=self.loop_matching_threshold, 
                distance_threshold=self.loop_matching_threshold
            )
            relative_pose = np2torch(relative_pose)
            match_idx_loop = corres_set[:, 0].tolist()
            match_idx_current = corres_set[:, 1].tolist()

        downsample_ids = downsample(match_idx_loop, self.downsample_num)
        if len(downsample_ids) <= 10:
            print(f"No matching Gaussians found, false loop.")
            return
        loop_edge = GaussianSLAMEdge(loop_submap_id, current_submap_id, relative_pose, self.loop_weight)
        print(f"Building loop constraint between submap_{loop_submap_id} and submap{current_submap_id}")
        self.add_edge(
            loop_vertex, current_vertex, loop_edge, 
            loop_gaussian_model.get_xyz()[loop_reused_pts_ids][match_idx_loop][downsample_ids], 
            loop_gaussian_model.get_scaling()[loop_reused_pts_ids][match_idx_loop][downsample_ids], 
            loop_gaussian_model.get_rotation()[loop_reused_pts_ids][match_idx_loop][downsample_ids], 
            SH2RGB(loop_gaussian_model.get_features()[loop_reused_pts_ids][match_idx_loop][downsample_ids]).clamp(0, 1),
            SH2RGB(current_gaussian_model.get_features()[current_reused_pts_ids][match_idx_current][downsample_ids]).clamp(0, 1)
        )
        print(f"Current error metric = {self.objective.error_metric()}")