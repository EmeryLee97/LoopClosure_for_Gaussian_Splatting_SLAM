import torch
import theseus as th
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from typing import Tuple, List, cast

from src.utils.pose_graph_utils import GaussianSLAMPoseGraph, GaussianSLAMEdge, match_gaussian_means


def create_data(
        num_pts: int = 100, 
        num_poses: int = 10, 
        translation_noise: float = 0.05, 
        rotation_noise: float = 0.1, 
        batch_size: int = 1
        ) -> Tuple[List[th.Point3], List[th.SE3], List[th.SE3], List[GaussianSLAMEdge]]:
    """
    create point clouds represented in different coordinates, record their ground truth 
    absolute pose, noisy absolute pose, also return an empty list to put loop edges

    Returns:
        point_list: a list stores points clouds, represented in different coordinates
        abs_pose_list_gt: a list stores ground truth absolute poses
        abs_pose_list: a list stores noisy (odometry) absolute poses
        edge_list: a list stores custum GaussianSLAMEdge
        TODO: Do I need to put the first edge that connets vertex_0 and vertex_1 into the list?
    """

    dtype = torch.float32 # will get error if changed to torch.float64, don't know why

    points_0 = th.Point3(2*torch.rand(num_pts, 3)-1, name="POINT_CLOUD__0") # initial points in world frame
    point_list = [points_0] # represented in different frames
    abs_pose_list_gt = [] # frame i to world frame
    abs_pose_list = [] # frame i to world frame (noisy)
    edge_list = []

    abs_pose_list_gt.append(th.SE3(
        tensor=torch.tile(torch.eye(3, 4, dtype=dtype), [1, 1, 1]),
        name="VERTEX_SE3_GT__0"
        ))
    
    abs_pose_list.append(th.SE3(
        tensor=torch.tile(torch.eye(3, 4, dtype=dtype), [1, 1, 1]),
        name="VERTEX_SE3__0"
        ))

    for idx in range(1, num_poses):

        # ground truth relative pose from frame_{idx-1} to frame_{idx}
        relative_pose_gt = th.SE3.exp_map(
            torch.cat([torch.rand(batch_size, 3)-0.5, 2.0 * torch.rand(batch_size, 3)-1], dim=1),
        )

        # generate points represented in frame_{idx}
        points = relative_pose_gt.transform_from(point_list[-1])
        points.name = f"POINT_CLOUD__{idx}"
        point_list.append(points)

        # add noise to get odometry relative pose from frame_{idx-1} to frame_{idx}
        relative_pose_noise = th.SE3.exp_map(
            torch.cat([
                translation_noise * (2.0 * torch.rand(batch_size, 3) - 1),
                rotation_noise * (2.0 * torch.rand(batch_size, 3) - 1),
            ] ,dim=1),
        )

        relative_pose = cast(th.SE3, relative_pose_noise.compose(relative_pose_gt))
        relative_pose.name = f"EDGE_SE3__{idx-1}_{idx}"
        weight = th.ScaleCostWeight(1.0, name=f"EDGE_WEIGHT__{idx-1}_{idx}")

        # absolute pose of frame_{idx}
        absolute_pose_gt = cast(th.SE3, abs_pose_list_gt[-1].compose(relative_pose_gt.inverse()))
        absolute_pose_gt.name = f"VERTEX_SE3_GT__{idx}"

        absolute_pose = cast(th.SE3, abs_pose_list[-1].compose(relative_pose.inverse()))
        absolute_pose.name = f"VERTEX_SE3__{idx}"

        abs_pose_list_gt.append(absolute_pose_gt)
        abs_pose_list.append(absolute_pose)

        # construct odometry edge between vertex_{idx-1} and vertex_{idx}
        edge_list.append(GaussianSLAMEdge(idx-1, idx, relative_pose, weight))

    return point_list, abs_pose_list_gt, abs_pose_list, edge_list


def add_loop_data(
        i: int, 
        j: int, 
        abs_pose_list_gt: List[th.SE3], 
        edge_list: List[GaussianSLAMEdge],
        coefficient: float = 1.0,
        measurement_noise:float = 0.01,
        batch_size: int = 1
        ) -> None:
    """
    Add loop closure between two arbitray coordinates i and j (i < j), and stores generated edge
    """

    if i >= j:
        raise ValueError(f"The first frame index {i} is greater than the second frame index {j}!")

    abs_pose_i_gt = abs_pose_list_gt[i]
    abs_pose_j_gt = abs_pose_list_gt[j]
    rel_pose_ij_gt = th.SE3.compose(abs_pose_j_gt.inverse(), abs_pose_i_gt)
    rel_pose_ij_gt.name = f"EDGE_SE3_GT__{i}_{j}"

    relative_pose_noise = th.SE3.exp_map(
            torch.cat([
                measurement_noise * (2.0 * torch.rand(batch_size, 3) - 1),
                measurement_noise * (2.0 * torch.rand(batch_size, 3) - 1),
            ] ,dim=1),
            )
    rel_pose_ij = cast(th.SE3, rel_pose_ij_gt.compose(relative_pose_noise))
    rel_pose_ij.name = f"EDGE_SE3__{i}_{j}"

    cost_weight = th.ScaleCostWeight(coefficient, name=f"EDGE_WEIGHT__{i}_{j}")
    edge = GaussianSLAMEdge(i, j, rel_pose_ij, cost_weight)
    edge_list.append(edge)


def draw_point_sets(point_list, abs_pose) -> None:

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Scatter Plot of Multiple Point Sets')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for i, points in enumerate(point_list):
        points_wf = (abs_pose[i][:3, :3] @ points.unsqueeze(-1)).squeeze() + abs_pose[i][:3, 3]
        ax.scatter(points_wf[:, 0], points_wf[:, 1], points_wf[:, 2], label=f'Set {i}')
    ax.legend()


if __name__ == "__main__":

    point_list, abs_pose_gt_list, abs_pose_list, edge_list = create_data()
    add_loop_data(1, 9, abs_pose_gt_list, edge_list)

    print(f"vertex_1 before optimization: {abs_pose_list[1]}")
    print(f"vertex_2 before optimization: {abs_pose_list[2]}")
    print(f"vertex_3 before optimization: {abs_pose_list[3]}")
    print(f"vertex_4 before optimization: {abs_pose_list[4]}")
    print(f"vertex_5 before optimization: {abs_pose_list[5]}")
    print(f"vertex_6 before optimization: {abs_pose_list[6]}")
    print(f"vertex_7 before optimization: {abs_pose_list[7]}")
    print(f"vertex_8 before optimization: {abs_pose_list[8]}")
    print(f"vertex_9 before optimization: {abs_pose_list[9]}")

    print("Constructing a pose graph for Gaussian Splatting SLAM.")
    pose_graph = GaussianSLAMPoseGraph(requires_auto_grad=True)

    for idx in range(len(edge_list)):
        edge = edge_list[idx]
        vertex_idx_i = edge.vertex_idx_i
        vertex_idx_j = edge.vertex_idx_j
        
        vertex_i = abs_pose_list[vertex_idx_i]
        vertex_j = abs_pose_list[vertex_idx_j]

        if vertex_idx_j - vertex_idx_i == 1:
            print(f"adding edge {idx} to pose graph, current edge is an odometry edge.")
            pose_graph.add_odometry_edge(vertex_i, vertex_j, edge, point_list[idx])
        else:
            print(f"adding edge {idx} to pose graph, current edge is an loop edge.")
            pose_i_gt, pose_j_gt = abs_pose_gt_list[vertex_idx_i], abs_pose_gt_list[vertex_idx_j]
            relative_pose_gt = th.SE3.compose(pose_j_gt.inverse(), pose_i_gt)
            inlier_idx = match_gaussian_means(point_list[vertex_idx_i].tensor, point_list[vertex_idx_j].tensor, relative_pose_gt.to_matrix().squeeze(), epsilon=5e-2)
            inlier_idx_i = [idx[0] for idx in inlier_idx]
            pose_graph.add_loop_closure_edge(vertex_i, vertex_j, edge, point_list[vertex_idx_i][inlier_idx_i], 1.0)

    info = pose_graph.optimize(max_iterations=1e5, step_size=0.01, verbose=False)
    print(info)

    print(f"vertex_1 ground truth: {abs_pose_gt_list[1]}")
    print(f"vertex_2 ground truth: {abs_pose_gt_list[2]}")
    print(f"vertex_3 ground truth: {abs_pose_gt_list[3]}")
    print(f"vertex_4 ground truth: {abs_pose_gt_list[4]}")
    print(f"vertex_5 ground truth: {abs_pose_gt_list[5]}")
    print(f"vertex_6 ground truth: {abs_pose_gt_list[6]}")
    print(f"vertex_7 ground truth: {abs_pose_gt_list[7]}")
    print(f"vertex_8 ground truth: {abs_pose_gt_list[8]}")
    print(f"vertex_9 ground truth: {abs_pose_gt_list[9]}")

