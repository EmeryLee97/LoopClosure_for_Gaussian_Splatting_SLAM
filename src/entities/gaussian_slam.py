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
from scipy.spatial.transform import Rotation as R

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.entities.logger import Logger
from src.entities.loop_closure import LoopClosureDetector
from src.entities.pose_graph import GaussianSLAMPoseGraph

from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.mapper_utils import exceeds_motion_thresholds
from src.utils.utils import np2torch, setup_seed, torch2np, get_id_from_string
from src.utils.vis_utils import *  # noqa - needed for debugging
from src.utils.io_utils import load_gaussian_from_submap_ckpt
from src.utils.pose_graph_utils import quaternion_multiplication

from src.evaluation.evaluate_trajectory import pose_error

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

    def pose_graph_optimization(self, frame_id, current_gaussian_model: GaussianModel) -> None:
        """ When a submap is finished, pgo should be trigered in 3 steps:
        1. add odometry constraint between the current and the last global keyframes to pose graph
        2. add loop constraint between the current and several past global keyframes to pose graph
        3. optimize the pose graph with LM algorithm
        """
        netvlad_feature = self.loop_closure_detector.get_netvlad_feature(self.dataset[self.new_submap_frame_ids[self.submap_id]][1])
        if self.submap_id > 0:
            last_gaussian_model, _ = load_gaussian_from_submap_ckpt(self.submap_id-1, self.output_path, self.opt)
            self.pose_graph.create_odometry_constraint(
                current_gaussian_model, last_gaussian_model, self.submap_id, self.new_submap_frame_ids, self.estimated_c2ws
            )
            if self.submap_id > 1:
                min_score = self.local_feature_index.get_min_score(netvlad_feature=netvlad_feature)
                print(f"Minimum score of submap_{self.submap_id} = {min_score}")
                self.local_feature_index.reset()
                _, loop_idx_list = self.loop_closure_detector.detect_knn(netvlad_feature=netvlad_feature, filter_threshold=min_score)
                for loop_idx in loop_idx_list:
                    loop_gaussian_model, _ = load_gaussian_from_submap_ckpt(loop_idx, self.output_path, self.opt)
                    self.pose_graph.create_loop_constraint(
                        current_gaussian_model, loop_gaussian_model, loop_idx, self.submap_id, self.new_submap_frame_ids, self.estimated_c2ws
                    )
                    #----------------------------------------------------------------------------------------
                    # self.pose_graph.logger.vis_submaps_overlap(
                    #     loop_gaussian_model, torch.eye(4, device='cuda'), loop_idx,
                    #     current_gaussian_model, torch.eye(4, device='cuda'), self.submap_id,
                    #     self.output_path / "blender_before"
                    # )
                    #----------------------------------------------------------------------------------------
                if len(loop_idx_list) != 0:
                    optimize_info = self.pose_graph.optimize()
                    print(optimize_info)
                    update_dict = {}
                    gt_poses = np.array(self.dataset.poses[:frame_id])
                    est_poses = torch2np(self.estimated_c2ws[:frame_id])
                    print(f"ATE_RMSE before: {pose_error(est_poses[:, :3, 3], gt_poses[:, :3, 3])['rmse']}")
                    #----------------------------------------------------------------------------------------
                    # for loop_idx in loop_idx_list:
                    #     loop_gaussian_model, _ = load_gaussian_from_submap_ckpt(loop_idx, self.output_path, self.opt)
                    #     if loop_idx == 0:
                    #         loop_vertex = self.pose_graph.objective.get_aux_var(f"VERTEX_SE3__{str(loop_idx).zfill(6)}")
                    #     else:
                    #         loop_vertex = self.pose_graph.objective.get_optim_var(f"VERTEX_SE3__{str(loop_idx).zfill(6)}")
                    #     current_vertex = self.pose_graph.objective.get_optim_var(f"VERTEX_SE3__{str(self.submap_id).zfill(6)}")
                        # self.pose_graph.logger.vis_submaps_overlap(
                        #     loop_gaussian_model, loop_vertex.tensor.squeeze().to('cuda'), loop_idx,
                        #     current_gaussian_model, current_vertex.tensor.squeeze().to('cuda'), self.submap_id,
                        #     self.output_path / "blender_after"
                        # )
                    #----------------------------------------------------------------------------------------
                    for pose_key, pose_val in optimize_info.best_solution.items():
                        submap_id = get_id_from_string(pose_key)
                        # modify the 3d Gaussians from checkpoints and save them again
                        pose_correction = torch.eye(4, device='cuda')
                        pose_correction[:3, :] = pose_val.squeeze().to('cuda')
                        quaternion_correction = R.from_matrix(torch2np(pose_correction[:3, :3])).as_quat()
                        quaternion_correction = np2torch(np.roll(quaternion_correction, 1)).to("cuda")
                        if submap_id == self.submap_id:
                            current_gaussian_model._xyz = current_gaussian_model._xyz @ pose_correction[:3, :3].transpose(-1, -2) + pose_correction[:3, 3].unsqueeze(-2)
                            current_gaussian_model._xyz = current_gaussian_model._xyz.detach()
                            current_gaussian_model._rotation = quaternion_multiplication(quaternion_correction, current_gaussian_model.get_rotation())
                            current_gaussian_model._rotation = current_gaussian_model._rotation.detach()
                            submap_start_idx = self.new_submap_frame_ids[submap_id]
                            submap_end_idx = frame_id
                        else:
                            gaussian_model_prev, submap_keyframes = load_gaussian_from_submap_ckpt(submap_id, self.output_path, self.opt)
                            gaussian_model_prev._xyz = gaussian_model_prev._xyz @ pose_correction[:3, :3].transpose(-1, -2) + pose_correction[:3, 3].unsqueeze(-2)
                            gaussian_model_prev._rotation = quaternion_multiplication(quaternion_correction, gaussian_model_prev.get_rotation())
                            gaussian_params = gaussian_model_prev.capture_dict()
                            submap_start_idx = self.new_submap_frame_ids[submap_id]
                            submap_end_idx = self.new_submap_frame_ids[submap_id+1]
                            submap_ckpt = {
                                "gaussian_params": gaussian_params,
                                "submap_keyframes": submap_keyframes,
                            }
                            save_dict_to_ckpt(
                                submap_ckpt, f"{str(submap_id).zfill(6)}.ckpt", directory=self.output_path / "submaps")
                            # TODO: torch.cuda.empty_cache()?
                            del gaussian_model_prev

                        for frame_idx in range(submap_start_idx, submap_end_idx):
                            pose_correction = pose_correction.to(self.estimated_c2ws[frame_idx].device)
                            self.estimated_c2ws[frame_idx] = pose_correction @ self.estimated_c2ws[frame_idx]
                        # reinitialize for next optimization
                        update_dict[pose_key] = torch.eye(3, 4, device='cuda').unsqueeze(0)
                    self.pose_graph.objective.update(update_dict)
                    # compare current ATE-RMSE
                    corr_poses = torch2np(self.estimated_c2ws[:frame_id])
                    print(f"ATE_RMSE after: {pose_error(corr_poses[:, :3, 3], gt_poses[:, :3, 3])['rmse']}")
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
                if self.optimize_with_loop_closure:
                    self.pose_graph_optimization(frame_id, gaussian_model)
                    if self.dataset_name in ["tum_rgbd", "scan_net"]:
                        print("Pose graph optimization triggered, retracking the current global keyframe.")
                        estimated_c2w = self.tracker.track(
                            frame_id, gaussian_model,
                            torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]), retrack=False)
                        self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)
                    
                save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
                gaussian_model = self.start_new_submap(frame_id, gaussian_model) # self.submap+=1 happens here, put everything before

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
                if self.optimize_with_loop_closure and self.submap_id > 1:
                    self.local_feature_index.add_to_index(self.dataset[frame_id][1])

        save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)

