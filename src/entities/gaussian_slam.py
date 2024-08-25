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

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.entities.logger import Logger
from src.entities.loop_closure import LoopClosureDetector
from src.entities.pose_graph import GaussianSLAMPoseGraph, GaussianSLAMEdge

from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.mapper_utils import exceeds_motion_thresholds
from src.utils.utils import np2torch, setup_seed, torch2np, get_id_from_string
from src.utils.vis_utils import *  # noqa - needed for debugging
from src.utils.io_utils import load_submap_ckpt, load_gaussian_from_submap_ckpt

class GaussianSLAM(object):

    def __init__(self, config: dict) -> None:

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})
        self.optimize_with_loop_closure = config["optimize_with_loop_closure"]

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

        if self.optimize_with_loop_closure:
            self.loop_closure_detector = LoopClosureDetector(config["loop_closure"])
            # stores the netvlad features of local keyframes in each submap
            self.local_feature_index = LoopClosureDetector(config["loop_closure"]) 
            self.pose_graph = GaussianSLAMPoseGraph(config["pose_graph"], self.dataset, self.logger)
            print('Loop closure detector config')  
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

    def pose_graph_optimization(self, current_gaussian_model, odometry_weight=1.0, loop_weight=1.0) -> None:
        """ When a submap is finished, pgo should be trigered in 3 steps:
        1. add odometry constraint between the current and the last global keyframes to pose graph
        2. add loop constraint between the current and several past global keyframes to pose graph
        3. optimize the pose graph with LM algorithm
        """
        netvlad_feature = self.loop_closure_detector.get_netvlad_feature(self.dataset[self.new_submap_frame_ids[self.submap_id]][1])
        if self.submap_id > 0:
            last_gaussian_model, _, _ = load_gaussian_from_submap_ckpt(self.submap_id-1, self.output_path, self.opt)
            self.pose_graph.create_odometry_constraint(
                current_gaussian_model, last_gaussian_model, self.new_submap_frame_ids, self.estimated_c2ws, odometry_weight
            )
            if self.submap_id > 1:
                min_score = self.local_feature_index.get_min_score(netvlad_feature=netvlad_feature)
                print(f"Minimum score of submap_{self.submap_id} = {min_score}")
                self.local_feature_index.reset()
                _, loop_idx_list = self.loop_closure_detector.detect_knn(netvlad_feature=netvlad_feature, filter_threshold=min_score)
                for loop_idx in loop_idx_list:
                    loop_gaussian_model, _, _ = load_gaussian_from_submap_ckpt(loop_idx, self.output_path, self.opt)
                    self.pose_graph.create_loop_constraint(
                        current_gaussian_model, loop_gaussian_model, loop_idx, self.new_submap_frame_ids, self.estimated_c2ws, loop_weight
                    )
                if len(loop_idx_list) != 0:
                    optimize_info = self.pose_graph.optimize()
                    update_dict = {}
                    for i, (pose_key, pose_val) in enumerate(optimize_info.best_solution.items()):
                        # modify the 3d Gaussians from checkpoints and save them again
                        pose_val = pose_val.squeeze().to('cuda')
                        gaussian_model_prev, submap_start_idx, submap_end_idx = load_gaussian_from_submap_ckpt(i+1, self.output_path, self.opt)
                        gaussian_model_prev._xyz = gaussian_model_prev._xyz @ pose_val[:3, :3].transpose(-1, -2) + pose_val[:3, 3].unsqueeze(-2)
                        # TODO: Do I also need to rotate the covariance?
                        gaussian_params = gaussian_model_prev.capture_dict()
                        submap_ckpt = {
                            "gaussian_params": gaussian_params,
                            "submap_keyframes": sorted(list(self.keyframes_info.keys()))
                        }
                        save_dict_to_ckpt(
                            submap_ckpt, f"{str(i+1).zfill(6)}.ckpt", directory=self.output_path / "submaps")
                        # TODO: torch.cuda.empty_cache()?
                        del gaussian_model_prev
                        # modify the poses in one submap TODO: interpolation?
                        pose_val.to(self.estimated_c2ws[frame_idx].device)
                        for frame_idx in range(submap_start_idx, submap_end_idx+1):
                            self.estimated_c2ws[frame_idx] = pose_val[:3, :3] @ self.estimated_c2ws[frame_idx] + pose_val[:3, 3]
                        # reinitialize for next optimization
                        update_dict[pose_key] = torch.eye(3, 4, device='cuda').unsqueeze(0)
                    self.pose_graph.objective.update(update_dict)
        self.loop_closure_detector.add_to_index(netvlad_feature=netvlad_feature)


    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0

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
                """ loop closure detection, pose graph optimizaiton, correct the poses and gaussians, reset the faiss index """
                if self.optimize_with_loop_closure:
                    self.pose_graph_optimization(gaussian_model)

                save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
                # self.submap += 1 = happens here, put everything before
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
                # add the current local keyframe info the local faiss index
                if self.submap_id > 1:
                    self.local_feature_index.add_to_index(self.dataset[frame_id][1])

        self.pose_graph_optimization(gaussian_model)
        save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)

