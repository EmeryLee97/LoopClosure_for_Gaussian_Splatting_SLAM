""" This module includes the Gaussian-SLAM class, which is responsible for controlling Mapper and Tracker
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import theseus as th

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.entities.logger import Logger

from src.utils.pose_graph_utils import GaussianSLAMPoseGraph, GaussianSLAMEdge, compute_relative_pose
from src.utils.loop_closure_utils import LoopClosureDetector
from src.utils.gaussian_model_utils import build_scaling_rotation
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.mapper_utils import exceeds_motion_thresholds
from src.utils.utils import np2torch, setup_seed, torch2np, get_id_from_string
from src.utils.vis_utils import *  # noqa - needed for debugging


class GaussianSLAM(object):

    def __init__(self, config: dict) -> None:

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})

        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        self.estimated_c2ws = torch.empty(n_frames, 4, 4)
        self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        self.submap_using_motion_heuristic = config["mapping"]["submap_using_motion_heuristic"]

        self.keyframes_info = {}
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids = [0]
        else:
            self.new_submap_frame_ids = frame_ids[::config["mapping"]["new_submap_every"]] + [n_frames - 1]
            self.new_submap_frame_ids.pop(0)

        self.logger = Logger(self.output_path, config["use_wandb"])
        self.mapper = Mapper(config["mapping"], self.dataset, self.logger)
        self.tracker = Tracker(config["tracking"], self.dataset, self.logger)

        print('Tracking config')
        pprint.PrettyPrinter().pprint(config["tracking"])
        print('Mapping config')
        pprint.PrettyPrinter().pprint(config["mapping"])      

        # TODO: add ckpt_netvlad and other params to config
        self.optimize_with_loop_closure = config["optimize_with_loop_closure"]
        if self.optimize_with_loop_closure:
            self.loop_closure_detector = LoopClosureDetector(config["loop_closure"])
            self.pose_graph = GaussianSLAMPoseGraph(config["pose_graph"])
            print('Loop closure config')  
            pprint.PrettyPrinter().pprint(config["loop_closure"])
            print('Pose graph config')
            pprint.PrettyPrinter().pprint(config["pose_graph"])
        

    def _setup_output_path(self, config: dict) -> None:
        """ Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        if "output_path" not in config["data"]:
            output_path = Path(config["data"]["output_path"])
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = output_path / self.timestamp
        else:
            self.output_path = Path(config["data"]["output_path"])
        self.output_path.mkdir(exist_ok=True, parents=True)
        os.makedirs(self.output_path / "mapping_vis", exist_ok=True)
        os.makedirs(self.output_path / "tracking_vis", exist_ok=True)

    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started based on the motion heuristic or specific frame IDs.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.
        """
        if self.submap_using_motion_heuristic:
            if exceeds_motion_thresholds(
                self.estimated_c2ws[frame_id], self.estimated_c2ws[self.new_submap_frame_ids[-1]],
                    rot_thre=50, trans_thre=0.5):
                return True
        elif frame_id in self.new_submap_frame_ids:
            return True
        return False

    def start_new_submap(self, frame_id: int, gaussian_model: GaussianModel) -> None:
        """ Initializes a new submap, saving the current submap's checkpoint and resetting the Gaussian model.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
            gaussian_model: The current GaussianModel instance to capture and reset for the new submap.
        Returns:
            A new, reset GaussianModel instance for the new submap.
        """
        # save the current sub-map as check point
        gaussian_params = gaussian_model.capture_dict()
        submap_ckpt_name = str(self.submap_id).zfill(6)
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(list(self.keyframes_info.keys()))
        }
        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
        
        # initialize new sub-map
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.mapper.keyframes = []
        self.keyframes_info = {}
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids.append(frame_id)
            self.mapping_frame_ids.append(frame_id)
        self.submap_id += 1
        return gaussian_model
    
    # -----------------------------------------------------------
    def load_gaussian_ckpt(self, submap_id, checkpoint_dir=None) -> torch.Tensor:
        """ Load saved sabmap from given checkpoint path"""
        #(self.output_path / "submaps" / f"{id.zfill(6)}.ckpt")
        if checkpoint_dir is None:
            checkpoint_dir = Path(self.output_path, "submaps")
        checkpoint_path = Path(checkpoint_dir, str(submap_id).zfill(6)+'.ckpt')
        submap = torch.load(checkpoint_path, map_location=self.device)
        #gaussian_model = GaussianModel()
        #gaussian_model.training_setup(self.opt)
        #gaussian_model.restore_from_params(submap["gaussian_params"], self.opt)
        #return gaussian_model
        xyz = submap["gaussian_params"]["xyz"]
        # scaling = submap["gaussian_params"]["scaling"]
        # rotation = submap["gaussian_params"]["rotation"]
        # covariance = build_scaling_rotation(scaling, rotation)
        return xyz
    
    def create_odometry_constraint(self, gaussian_model_current: GaussianModel, cost_weight: float) -> None:
        current_submap_id = self.new_submap_frame_ids[-1]
        current_submap_pose = self.estimated_c2ws[current_submap_id]
        last_submap_id = self.new_submap_frame_ids[-2]
        last_submap_pose = self.estimated_c2ws[last_submap_id]
        relative_pose_odometry = current_submap_pose.inverse() @ last_submap_pose
        odometry_edge = GaussianSLAMEdge(last_submap_id, current_submap_id, relative_pose_odometry,cost_weight)
        # A theseus variable can not be recreated with the same name, and note that i < j
        if self.pose_graph.objective.has_optim_var(f"VERTEX_SE3__{last_submap_id}"):
            vertex_i = self.pose_graph.objective.get_optim_var(f"VERTEX_SE3__{last_submap_id}")
        else:
            vertex_i = th.SE3(tensor=last_submap_pose, name=f"VERTEX_SE3__{last_submap_id}")
        # For an odometry edge, the second vertex is always newly created
        vertex_j=th.SE3(tensor=current_submap_pose, name=f"VERTEX_SE3__{current_submap_id}")
        self.pose_graph.add_odometry_edge(vertex_i, vertex_j, odometry_edge, gaussian_model_current.get_xyz())

    def create_loop_constraint(self, gaussian_model_current: GaussianModel, cost_weight: float) -> None:
        current_submap_id = self.new_submap_frame_ids[-1]
        current_submap_rgb = self.dataset[current_submap_id][1]
        current_submap_depth = self.dataset[current_submap_id][2]
        current_submap_pose = self.estimated_c2ws[current_submap_id]
        vertex_j = self.pose_graph.objective.get_optim_var(f"VERTEX_SE3__{current_submap_id}")
        
        loop_submap_id_list = self.loop_closure_detector.detect_knn(
            np2torch(current_submap_rgb, 'cuda'), 
            add_to_index=True
        )
        for loop_submap_ordinal in loop_submap_id_list:
            loop_submap_id = self.new_submap_frame_ids[loop_submap_ordinal]
            loop_submap_depth = self.dataset[loop_submap_id][2]
            loop_submap_pose = self.estimated_c2ws[loop_submap_id]
            gaussian_model_loop = self.load_gaussian_ckpt(loop_submap_id, Path(self.output_path, "submaps"))      
            relative_pose_measurement = compute_relative_pose(
                loop_submap_depth,
                loop_submap_pose,
                gaussian_model_loop,
                current_submap_depth,
                current_submap_pose,
                gaussian_model_current,
                self.dataset.intrinsics,
                voxel_size=0.5 # TODO: move this to config
            )
            matching_idx = self.pose_graph.match_gaussian_means(
                gaussian_model_loop, 
                gaussian_model_current,
                relative_pose_measurement,
            )
            matching_idx_current = [idx_pair[1] for idx_pair in matching_idx]
            vertex_i = self.pose_graph.objective.get_optim_var(f"VERTEX_SE3__{loop_submap_id}")
            loop_edge = GaussianSLAMEdge(loop_submap_id, current_submap_id, relative_pose_measurement, cost_weight)
            self.pose_graph.add_loop_closure_edge(
                vertex_i, 
                vertex_j, 
                loop_edge, 
                gaussian_model_current.get_xyz()[matching_idx_current]
            )

    def pose_graph_optimization(self, next_submap_id: int, gaussian_model_current: GaussianModel, cost_weight: float) -> None:
        """ When a submap is finished, pgo should be trigered in 3 steps:
        1. add odometry constraint between the current and the last global keyframes to pose graph
        2. add loop constraint between the current and several past global keyframes to pose graph
           a. detect k nearest rgb frames (except for the last one)
           b. calculate relative poses between the current global keyframes and those detected ones
           c. add these loop constraints to pose graph
        3. optimize the pose graph with LM algorithm
        """
        current_submap_id = self.new_submap_frame_ids[-1]
        current_submap_rgb = self.dataset[current_submap_id][1]
        # For the first submap, only add rgb image to faiss index
        if len(self.new_submap_frame_ids) == 1:
            self.loop_closure_detector.add_to_index(np2torch(current_submap_rgb, 'cuda'))
        # For the second submap, add odometry constraint to pose graph, add rgb image to faiss index
        elif len(self.new_submap_frame_ids) == 2:
            self.create_odometry_constraint(gaussian_model_current, cost_weight)
            self.loop_closure_detector.add_to_index(np2torch(current_submap_rgb, 'cuda'))
        # For the other submaps, add odometry and loop constraints to pose graph, add rgb images to faiss index
        else:
            self.create_odometry_constraint(gaussian_model_current, cost_weight)
            self.create_loop_constraint(gaussian_model_current, cost_weight)
            optimize_info = self.pose_graph.optimize_two_steps()  
            best_solution = optimize_info.best_solution

            correct_mat = best_solution[f"VERTEX_SE3__000000"].inverse()
            for pose_key, pose_val in best_solution.items():
                if "VERTEX_SE3" in pose_key:
                    submap_id = get_id_from_string(pose_val)
                    submap_index = self.new_submap_frame_ids.index(submap_id)
                    if submap_id == self.new_submap_frame_ids[-1]: # if is the current submap
                        last_frame_id = next_submap_id
                    else:
                        last_frame_id = self.new_submap_frame_ids[submap_index+1]

                    for frame_id in range(submap_id, last_frame_id):
                        self.estimated_c2ws[frame_id] = correct_mat @ self.estimated_c2ws[frame_id]
                    
    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0
        #self.loop_closure_detector.add_to_index(self.dataset[0][1])

        for frame_id in range(len(self.dataset)):

            if frame_id in [0, 1]:
                estimated_c2w = self.dataset[frame_id][-1]
            else:
                estimated_c2w = self.tracker.track(
                    frame_id, gaussian_model,
                    torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))
            self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

            # Reinitialize gaussian model for new segment
            if self.should_start_new_submap(frame_id):
                if self.optimize_with_loop_closure:
                    print(f"Optimizing with loop closure, currently {len(self.new_submap_frame_ids)} submaps.")
                    self.pose_graph_optimization(frame_id, gaussian_model, 1.0)
                    
                save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
                gaussian_model = self.start_new_submap(frame_id, gaussian_model)

            if frame_id in self.mapping_frame_ids:
                print("\nMapping frame", frame_id)
                gaussian_model.training_setup(self.opt)
                estimate_c2w = torch2np(self.estimated_c2ws[frame_id])
                new_submap = not bool(self.keyframes_info)
                opt_dict = self.mapper.map(frame_id, estimate_c2w, gaussian_model, new_submap)

                # Keyframes info update
                self.keyframes_info[frame_id] = {
                    "keyframe_id": len(self.keyframes_info.keys()),
                    "opt_dict": opt_dict
                }
        save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)

