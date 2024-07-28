""" This module includes the GaussianSLAMEdge class and the GaussianSLAMPoseGraph class """

import numpy as np
import torch
import theseus as th
from typing import List, Dict, Type

from src.entities.datasets import BaseDataset
from src.entities.gaussian_model import GaussianModel
from src.utils.mapper_utils import compute_camera_frustum_corners, compute_frustum_point_ids
from src.utils.utils import torch2np, np2torch, np2ptcloud
from src.utils.io_utils import load_gaussian_ckpt
from src.utils.pose_graph_utils import error_fn_dense_surface_alignment, error_fn_line_process, match_gaussian_means, \
                                        downsample, point_cloud_registration


class GaussianSLAMEdge:
    def __init__(self, vertex_idx_i: int, vertex_idx_j: int, relative_pose: torch.Tensor, cost_weight=1.0, device='cpu'):
        # be careful of data type (torch.float64 and torch.float32)
        self.device = device
        self.vertex_idx_i = vertex_idx_i
        self.vertex_idx_j = vertex_idx_j
        self.relative_pose = th.SE3(
            tensor=relative_pose.squeeze().unsqueeze(0)[:, 3, :], 
            name=f"EDGE_SE3__{str(vertex_idx_i).zfill(6)}_{str(vertex_idx_j).zfill(6)}"
        )
        self.relative_pose.to(self.device)
        self.cost_weight = th.ScaleCostWeight(
            scale=cost_weight,
            name=f"EDGE_WEIGHT__{str(vertex_idx_i).zfill(6)}_{str(vertex_idx_j).zfill(6)}"
        )
        self.cost_weight.to(self.device)


class GaussianSLAMPoseGraph:
    def __init__(self, config: Dict, device='cpu', requires_auto_grad=True):
        self._requires_auto_grad = requires_auto_grad
        self.device = device
        self.objective = th.Objective()
        self.objective.to(self.device)

        self.use_gt_relative_pose = config["use_gt_relative_pose"]
        self.center_matching_threshold = config["center_matching_threshold"]
        self.optimization_max_iterations = config["optimization_max_iterations"]
        self.optimization_step_size = config["optimization_step_size"]
        self.loop_inlier_threshold = config["loop_inlier_threshold"]
        self.correspondence_factor = config["correspondence_factor"]
        self.damping = config["damping"]
        self.track_best_solution = config["track_best_solution"]
        self.verbose = config["verbose"]

    def add_odometry_edge(self,
            vertex_i: th.SE3,
            vertex_j: th.SE3,
            edge: GaussianSLAMEdge,
            gaussian_means: torch.Tensor # should have shape (num_pts, 3)
        ) -> None:
        """ add an odometry constraint to the objective"""
        gaussian_means_th = th.Variable(
            tensor=gaussian_means.unsqueeze(0), 
            name=f"GAUSSIAN_MEANS_ODOMETRY__{str(edge.vertex_idx_i).zfill(6)}_{str(edge.vertex_idx_j).zfill(6)}"
        )
        vertex_i.to(self.device)
        vertex_j.to(self.device)
        gaussian_means_th.to(self.device)

        if self._requires_auto_grad:
            cost_fn = th.AutoDiffCostFunction(
                optim_vars=[vertex_i, vertex_j],
                err_fn=error_fn_dense_surface_alignment, 
                dim=gaussian_means.shape[0], 
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
            gaussian_means: torch.Tensor, # matching inliers
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
            name=f"GAUSSIAN_MEANS_ODOMETRY__{str(edge.vertex_idx_i).zfill(6)}_{str(edge.vertex_idx_j).zfill(6)}")
        
        vertex_i.to(self.device)
        vertex_j.to(self.device)
        cost_weight_line_process.to(self.device)
        l_ij.to(self.device)
        gaussian_means_th.to(self.device)

        if self._requires_auto_grad:
            cost_fn_registration = th.AutoDiffCostFunction(
                optim_vars=[vertex_i, vertex_j, l_ij], 
                err_fn=error_fn_dense_surface_alignment, 
                dim=gaussian_means.shape[0],
                cost_weight=cost_weight_registration, 
                aux_vars=[edge.relative_pose, gaussian_means_th],
                name=f"COST_FUNCTION_LINE_PROCRSS__{str(edge.vertex_idx_i).zfill(6)}_{str(edge.vertex_idx_j).zfill(6)}"
            )
            self.objective.add(cost_fn_registration)
            cost_fn_line_process = th.AutoDiffCostFunction(
                optim_vars=[l_ij,], 
                err_fn=error_fn_line_process, 
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
            #abs_err_tolerance=1e-9,
            rel_err_tolerance=1e-4,
            vectorize=True,
        )
        layer = th.TheseusLayer(optimizer)
        layer.to(self.device)
        with torch.no_grad():
            _, info = layer.forward(optimizer_kwargs={
                "damping": self.damping, 
                "track_best_solution": self.track_best_solution, 
                "verbose": self.verbose
            })
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

    # TODO: 看看输入参数有没有在什么地方被改变, 删除占用内存过多的变量
    def create_odometry_constraint(
            self, 
            new_submap_frame_ids: List,
            estimated_c2ws: List,
            xyz_last: torch.Tensor,
            xyz_current: torch.Tensor, 
            dataset: Type[BaseDataset],
            cost_weight=1.0,
            downsample_num=500,
        ) -> None:
        """ create an odometry constraint between the last submap and the current submap, then add it into pose graph.
        Each vertex is initialized as identity matrix, with name 'VERTEX_SE3__str(i).zfill(6)', where 'i' indicates it's 
        the i'th submap. The relative pose between adjacent submaps is also initialized to identity matrix. 
        Args:
            new_submap_frame_ids: a list that stores the global keyframe id for each submap
            estimated_c2ws: a list that stores the estimated poses of each tracking frame
            xyz_last, xyz_current: point cloud positions
            dataset: the dataset that is working on
            cost_weight: weight for this odometry constraint
            downsample_num: downsample number
        """
        last_ordinal = len(new_submap_frame_ids) - 2
        last_submap_id = new_submap_frame_ids[-2]
        last_submap_pose = estimated_c2ws[last_submap_id]
        last_frustum_corners = compute_camera_frustum_corners(dataset[last_submap_id][2], last_submap_pose, dataset.intrinsics)
        last_reused_pts_ids = compute_frustum_point_ids(xyz_last, last_frustum_corners, device=self.device)
        if self.objective.has_optim_var(f"VERTEX_SE3__{str(last_ordinal).zfill(6)}"):
            last_vertex = self.objective.get_optim_var(f"VERTEX_SE3__{str(last_ordinal).zfill(6)}")
        else:
            last_vertex = th.SE3(tensor=torch.tile(torch.eye(3, 4), [1, 1, 1]), name=f"VERTEX_SE3__{str(last_ordinal).zfill(6)}")

        current_ordinal = len(new_submap_frame_ids) - 1
        current_submap_id = new_submap_frame_ids[-1]
        current_submap_pose = estimated_c2ws[current_submap_id]
        current_frustum_corners = compute_camera_frustum_corners(dataset[current_submap_id][2], current_submap_pose, dataset.intrinsics)
        current_reused_pts_ids = compute_frustum_point_ids(xyz_current, current_frustum_corners, device=self.device)
        if self.objective.has_optim_var(f"VERTEX_SE3__{str(current_ordinal).zfill(6)}"):
            current_vertex = self.objective.get_optim_var(f"VERTEX_SE3__{str(current_ordinal).zfill(6)}")
        else:
            current_vertex = th.SE3(tensor=torch.tile(torch.eye(3, 4), [1, 1, 1]), name=f"VERTEX_SE3__{str(current_ordinal).zfill(6)}")

        matching_ids = match_gaussian_means(
            xyz_last[last_reused_pts_ids], 
            xyz_current[current_reused_pts_ids], 
            torch.eye(4), 
            self.center_matching_threshold
        )
        downsample_ids = downsample(matching_ids, downsample_num)
        odometry_edge = GaussianSLAMEdge(last_ordinal, current_ordinal, torch.eye(3, 4), cost_weight)
        print(f"Building odometry constraint between submap_{last_ordinal} and submap{current_ordinal}")
        self.add_odometry_edge(last_vertex, current_vertex, odometry_edge, xyz_last[last_reused_pts_ids][downsample_ids])

    def create_loop_constraint(
        self, 
        loop_ids: List,
        new_submap_frame_ids: List,
        estimated_c2ws: List,
        xyz_current: torch.Tensor, 
        dataset: Type[BaseDataset],
        cost_weight=1.0,
        downsample_num=500,
    ) -> None:
        """ create a loop constraint between the current submap and a submap that forms a loop with the current one, then
        add it into pose graph. Each vertex is initialized as identity matrix, with name 'VERTEX_SE3__str(i).zfill(6)', 
        where 'i' indicates it's the i'th submap. The relative pose between those submaps is calculated. 
        Args:
            loop_ids: a list that contains index of submaps that form a loop with the current submap
            new_submap_frame_ids: a list that stores the global keyframe id for each submap
            estimated_c2ws: a list that stores the estimated poses of each tracking frame
            xyz_last, xyz_current: point cloud positions
            dataset: the dataset that is working on
            cost_weight: weight for this odometry constraint
            downsample_num: downsample number
        """
        # Note that vertex name is the submap ordinal (0, 1, 2, ...), not the keyframe id
        current_ordinal = len(new_submap_frame_ids) - 1
        current_submap_id = new_submap_frame_ids[-1]
        current_submap_pose = estimated_c2ws[current_submap_id]
        current_frustum_corners = compute_camera_frustum_corners(dataset[current_submap_id][2], current_submap_pose, dataset.intrinsics)
        current_reused_pts_ids = compute_frustum_point_ids(xyz_current, current_frustum_corners, device=self.device)
        if self.objective.has_optim_var(f"VERTEX_SE3__{str(current_ordinal).zfill(6)}"):
            current_vertex = self.objective.get_aux_var(f"VERTEX_SE3__{str(current_ordinal).zfill(6)}")
        else:
            current_vertex = th.SE3(tensor=torch.tile(torch.eye(3, 4), [1, 1, 1]), name=f"VERTEX_SE3__{str(current_ordinal).zfill(6)}")
        
        for loop_ordinal in loop_ids:
            loop_submap_id = new_submap_frame_ids[loop_ordinal]
            loop_submap_pose = estimated_c2ws[loop_submap_id]
            loop_frustum_corners = compute_camera_frustum_corners(dataset[loop_submap_id][2], loop_submap_pose, dataset.intrinsics)
            loop_reused_pts_ids = compute_frustum_point_ids(xyz_loop, loop_frustum_corners, device=self.device)
            if self.objective.has_optim_var(f"VERTEX_SE3__{str(loop_ordinal).zfill(6)}"):
                loop_vertex = self.objective.get_optim_var(f"VERTEX_SE3__{str(loop_ordinal).zfill(6)}")
            else:
                loop_vertex = th.SE3(tensor=torch.tile(torch.eye(3, 4), [1, 1, 1]), name=f"VERTEX_SE3__{str(loop_ordinal).zfill(6)}")
            loop_submap = load_gaussian_ckpt(loop_ordinal, ) # TODO
            xyz_loop = loop_submap['gaussian_params']['xyz']

            # TODO: type of poses?
            if self.use_gt_relative_pose:
                current_correction_pose_gt = np2torch(dataset[current_submap_id][-1] @ np.linalg.inv(current_submap_pose))
                loop_correction_pose = np2torch(dataset[loop_submap_id][-1] @ np.linalg.inv(loop_submap_pose))
                relative_pose_measurement = None
            else:
                relative_pose_measurement = point_cloud_registration(
                    np2ptcloud(torch2np(xyz_loop)),
                    np2ptcloud(torch2np(xyz_current))
                ) # TODO: implementation
            loop_edge = GaussianSLAMEdge(loop_ordinal, current_ordinal, relative_pose_measurement, cost_weight)
            matching_ids = match_gaussian_means(
                xyz_loop[loop_reused_pts_ids], 
                xyz_current[current_reused_pts_ids], 
                torch.eye(4), 
                self.center_matching_threshold
            )
            del xyz_loop
            # TODO: if too few matching points, pass this loop?
            downsample_ids = downsample(matching_ids, downsample_num)
            loop_edge = GaussianSLAMEdge(loop_ordinal, current_ordinal, relative_pose_measurement, cost_weight)
            self.add_loop_closure_edge(current_vertex, loop_vertex, loop_edge, xyz_current[current_reused_pts_ids][downsample_ids])