""" This module includes the Logger class, which is responsible for logging for both Mapper and the Tracker """
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy.spatial.transform import Rotation as R
import bpy
import blender_plots as bplt
from datetime import datetime
from PIL import ImageColor

from src.entities.gaussian_model import GaussianModel
from src.utils.gaussian_model_utils import SH2RGB


class Logger(object):

    def __init__(self, output_path: Union[Path, str], use_wandb=False) -> None:
        self.output_path = Path(output_path)
        (self.output_path / "mapping_vis").mkdir(exist_ok=True, parents=True)
        (self.output_path / "pose_graph_vis").mkdir(exist_ok=True, parents=True)
        self.use_wandb = use_wandb

    def log_tracking_iteration(self, frame_id, cur_pose, gt_quat, gt_trans, total_loss,
                               color_loss, depth_loss, iter, num_iters,
                               wandb_output=False, print_output=False) -> None:
        """ Logs tracking iteration metrics including pose error, losses, and optionally reports to Weights & Biases.
        Logs the error between the current pose estimate and ground truth quaternion and translation,
        as well as various loss metrics. Can output to wandb if enabled and specified, and print to console.
        Args:
            frame_id: Identifier for the current frame.
            cur_pose: The current estimated pose as a tensor (quaternion + translation).
            gt_quat: Ground truth quaternion.
            gt_trans: Ground truth translation.
            total_loss: Total computed loss for the current iteration.
            color_loss: Computed color loss for the current iteration.
            depth_loss: Computed depth loss for the current iteration.
            iter: The current iteration number.
            num_iters: The total number of iterations planned.
            wandb_output: Whether to output the log to wandb.
            print_output: Whether to print the log output.
        """

        quad_err = torch.abs(cur_pose[:4] - gt_quat).mean().item()
        trans_err = torch.abs(cur_pose[4:] - gt_trans).mean().item()
        if self.use_wandb and wandb_output:
            wandb.log(
                {
                    "Tracking/idx": frame_id,
                    "Tracking/cam_quad_err": quad_err,
                    "Tracking/cam_position_err": trans_err,
                    "Tracking/total_loss": total_loss.item(),
                    "Tracking/color_loss": color_loss.item(),
                    "Tracking/depth_loss": depth_loss.item(),
                    "Tracking/num_iters": num_iters,
                })
        if iter == num_iters - 1:
            msg = f"frame_id: {frame_id}, cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f} "
        else:
            msg = f"iter: {iter}, color_loss: {color_loss.item():.5f}, depth_loss: {depth_loss.item():.5f} "
        msg = msg + f", cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f}"
        if print_output:
            print(msg, flush=True)

    def log_mapping_iteration(self, frame_id, new_pts_num, model_size, iter_opt_time, opt_dict: dict) -> None:
        """ Logs mapping iteration metrics including the number of new points, model size, and optimization times,
        and optionally reports to Weights & Biases (wandb).
        Args:
            frame_id: Identifier for the current frame.
            new_pts_num: The number of new points added in the current mapping iteration.
            model_size: The total size of the model after the current mapping iteration.
            iter_opt_time: Time taken per optimization iteration.
            opt_dict: A dictionary containing optimization metrics such as PSNR, color loss, and depth loss.
        """
        if self.use_wandb:
            wandb.log({"Mapping/idx": frame_id,
                       "Mapping/num_total_gs": model_size,
                       "Mapping/num_new_gs": new_pts_num,
                       "Mapping/per_iteration_time": iter_opt_time,
                       "Mapping/psnr_render": opt_dict["psnr_render"],
                       "Mapping/color_loss": opt_dict[frame_id]["color_loss"],
                       "Mapping/depth_loss": opt_dict[frame_id]["depth_loss"]})

    def vis_mapping_iteration(self, frame_id, iter, color, depth, gt_color, gt_depth, seeding_mask=None) -> None:
        """
        Visualization of depth, color images and save to file.

        Args:
            frame_id (int): current frame index.
            iter (int): the iteration number.
            save_rendered_image (bool): whether to save the rgb image in separate folder
            img_dir (str): the directory to save the visualization.
            seeding_mask: used in mapper when adding gaussians, if not none.
        """
        gt_depth_np = gt_depth.cpu().numpy()
        gt_color_np = gt_color.cpu().numpy()

        depth_np = depth.detach().cpu().numpy()
        color = torch.round(color * 255.0) / 255.0
        color_np = color.detach().cpu().numpy()
        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        # make errors >=5cm noticeable
        depth_residual = np.clip(depth_residual, 0.0, 0.05)

        color_residual = np.abs(gt_color_np - color_np)
        color_residual[np.squeeze(gt_depth_np == 0.0)] = 0.0

        # Determine Aspect Ratio and Figure Size
        aspect_ratio = color.shape[1] / color.shape[0]
        fig_height = 8
        # Adjust the multiplier as needed for better spacing
        fig_width = fig_height * aspect_ratio * 1.2

        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        axs[0, 0].imshow(gt_depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 0].set_title('Input Depth', fontsize=16)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 1].set_title('Rendered Depth', fontsize=16)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap="plasma")
        axs[0, 2].set_title('Depth Residual', fontsize=16)
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title('Input RGB', fontsize=16)
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title('Rendered RGB', fontsize=16)
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        if seeding_mask is not None:
            axs[1, 2].imshow(seeding_mask, cmap="gray")
            axs[1, 2].set_title('Densification Mask', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
        else:
            axs[1, 2].imshow(color_residual, cmap="plasma")
            axs[1, 2].set_title('RGB Residual', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])

        for ax in axs.flatten():
            ax.axis('off')
        fig.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust top margin
        fig_name = str(self.output_path / "mapping_vis" / f'{frame_id:04d}_{iter:04d}.jpg')
        fig_title = f"Mapper Color/Depth at frame {frame_id:04d} iters {iter:04d}"
        plt.suptitle(fig_title, y=0.98, fontsize=20)
        plt.savefig(fig_name, dpi=250, bbox_inches='tight')
        plt.clf()
        plt.close()
        if self.use_wandb:
            log_title = "Mapping_vis/" + f'{frame_id:04d}_{iter:04d}'
            wandb.log({log_title: [wandb.Image(fig_name)]})
        print(f"Saved rendering vis of color/depth at {frame_id:04d}_{iter:04d}.jpg")

    # def log_pose_graph_iteration(self):
    #     if self.use_wandb:
    #         wandb.log


    def vis_submaps(self,
        gaussian_model_i: GaussianModel, pose_i: torch.Tensor, idx_i: int,
        gaussian_model_j: GaussianModel, pose_j: torch.Tensor, idx_j: int,
        output_path: Union[str, Path]
    ) -> None:
        """ Visualizes the overlapped Gaussians from two submaps """
        gaussian_xyz = torch.cat((
            gaussian_model_i.get_xyz() @ pose_i[:3, :3].transpose(-1, -2) + pose_i[:3, :3].unsqueeze(-2),
            gaussian_model_j.get_xyz() @ pose_j[:3, :3].transpose(-1, -2) + pose_j[:3, :3].unsqueeze(-2)),
            dim=-2
        ).detach().cpu().numpy()
        gaussian_scaling = torch.cat((gaussian_model_i.get_scaling(), gaussian_model_j.get_scaling()), dim=-2).detach().cpu().numpy()
        gaussian_rotation = R.from_quat(
            torch.cat((gaussian_model_i.get_rotation(), gaussian_model_j.get_rotation()), dim=-2).detach().cpu().numpy()[:, [1, 2, 3, 0]]).as_matrix()
        gaussian_color = SH2RGB(torch.cat((gaussian_model_i.get_features().squeeze(), gaussian_model_j.get_features().squeeze()), dim=-2)).clamp(0, 1)
        gaussian_opacity = torch.cat((gaussian_model_i.get_opacity(), gaussian_model_j.get_opacity()), dim=-2)
        gaussian_color_opaticy = torch.cat((gaussian_color, gaussian_opacity), dim=-1).detach().cpu().numpy()
            
        camera_location = [-0.778881, -1.854416, 3.641333]
        camera_rotation = [-5.921588, -0.000003, -12.637432]
        bplt.scene_utils.setup_scene(
            camera_location=camera_location,
            camera_rotation=camera_rotation,
            clear=True,
            sun_energy=1,
        )
        scatter = bplt.Scatter(
            gaussian_xyz,
            color=gaussian_color_opaticy,
            name=f'submaps_overlap_{idx_i}_{idx_j}',
            marker_scale=gaussian_scaling,
            marker_rotation=gaussian_rotation,
            marker_type='ico_spheres',
            subdivisions=2,
        )
        scatter.base_object.rotation_euler = [np.pi, 0, 0]
        bpy.context.scene.render.film_transparent = True
        if "Cube" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["Cube"])
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        bplt.scene_utils.render_image(
            f'scene{timestamp}.png',
            resolution=(1024 // 1, 600 // 1),
            samples=10,
        )
        save_path = Path(output_path, f'submap_overlap__{idx_i}_{idx_j}.blend')
        bpy.ops.wm.save_as_mainfile(filepath=str(save_path))


    def vis_submaps_overlap(self,
        gaussian_model_i: GaussianModel, pose_i: torch.Tensor, idx_i: int,
        gaussian_model_j: GaussianModel, pose_j: torch.Tensor, idx_j: int,
        output_path: Union[str, Path]
    ) -> None:
        """ Visualizes the overlapped Gaussians from two submaps """
        gaussian_xyz = torch.cat((
            gaussian_model_i.get_xyz() @ pose_i[:3, :3].transpose(-1, -2) + pose_i[:3, :3].unsqueeze(-2),
            gaussian_model_j.get_xyz() @ pose_j[:3, :3].transpose(-1, -2) + pose_j[:3, :3].unsqueeze(-2)),
            dim=-2
        ).detach().cpu().numpy()
        gaussian_scaling = torch.cat((gaussian_model_i.get_scaling(), gaussian_model_j.get_scaling()), dim=-2).detach().cpu().numpy()
        gaussian_rotation = R.from_quat(
        torch.cat((gaussian_model_i.get_rotation(), gaussian_model_j.get_rotation()), dim=-2).detach().cpu().numpy()[:, [1, 2, 3, 0]]).as_matrix()
        
            
        camera_location = [-0.778881, -1.854416, 3.641333]
        camera_rotation = [-5.921588, -0.000003, -12.637432]
        bplt.scene_utils.setup_scene(
            camera_location=camera_location,
            camera_rotation=camera_rotation,
            clear=True,
            sun_energy=1,
        )
        colors = [[c / 255 for c in ImageColor.getcolor(color, "RGB")]for color in plt.rcParams['axes.prop_cycle'].by_key()['color']]

        scatter = bplt.Scatter(
            gaussian_xyz,
            color=colors[2],
            name=f'submaps_overlap_{idx_i}_{idx_j}',
            marker_scale=gaussian_scaling,
            marker_rotation=gaussian_rotation,
            marker_type='ico_spheres',
            subdivisions=2,
        )
        scatter.base_object.rotation_euler = [np.pi, 0, 0]
        bpy.context.scene.render.film_transparent = True
        if "Cube" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["Cube"])
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        bplt.scene_utils.render_image(
            f'scene{timestamp}.png',
            resolution=(1024 // 1, 600 // 1),
            samples=10,
        )
        save_path = output_path / f'submap_overlap__{idx_i}_{idx_j}.blend'
        bpy.ops.wm.save_as_mainfile(filepath=str(save_path))

# TODO: what kind of visualizations do I want?