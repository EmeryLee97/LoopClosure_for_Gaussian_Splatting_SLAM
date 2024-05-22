import torch
import theseus as th
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from src.utils.pose_graph_utils import dense_surface_alignment, GaussianSLAMPoseGraph


def create_data(num_pts=100, time_steps=10, noise=0.1) -> None:
    """
    create 11 point clouds, record their ground truth absolute pose, noisy absolute pose,
    and also return an empty list to put loop edges
    """

    points_1 = 2 * torch.rand(num_pts, 3) - 1 # initial points in world frame
    point_list = [points_1] # represented in different frames
    abs_pose_gt = [torch.eye(4)] # frame 1 to world frame
    abs_pose_noisy = [torch.eye(4)] # frame 1 to world frame (noisy)
    rel_pose_dict_loop = {} # relative pose between loops

    for _ in range(time_steps):
        
        # ground truth relative pose from frame{idx} to frame{idx+1}
        rotation_gt = torch.Tensor(Rotation.from_euler('xyz', 10*torch.rand(3)-5, degrees=True).as_matrix()) # [-5, 5) degree
        
        translation_gt = 20*torch.rand(3)-10 # [-10, 10) meter

        relative_pose_gt = torch.eye(4) # ground truth pose
        relative_pose_gt[:3, :3] = rotation_gt
        relative_pose_gt[:3, 3] = translation_gt

        # generate points represented in frame{idx+1}
        points = (rotation_gt @ points_1.unsqueeze(dim=-1)).squeeze() + translation_gt
        point_list.append(points)

        # add noise to get odometry relative pose from frame{idx} to frame{idx+1}
        noise_rot = torch.Tensor(Rotation.from_euler('xyz', noise*(10*torch.randn(3)-5), degrees=True).as_matrix())
        rotation_noisy = noise_rot @ rotation_gt

        noise_trans = noise * 5 * (2*torch.randn(3)-1)
        translation_noisy = translation_gt + noise_trans

        relative_pose_noisy = torch.eye(4) # odometry pose
        relative_pose_noisy[:3, :3] = rotation_noisy
        relative_pose_noisy[:3, 3] = translation_noisy
        
        abs_pose_gt.append(abs_pose_gt[-1] @ torch.linalg.inv(relative_pose_gt))
        abs_pose_noisy.append(abs_pose_noisy[-1] @ torch.linalg.inv(relative_pose_noisy))

    return point_list, abs_pose_gt, abs_pose_noisy, rel_pose_dict_loop


def add_loop_closure_constraint(abs_pose_gt, rel_pose_dict_loop, idx_i, idx_j, measurement_noise=0.2) -> None:

    if idx_j >= len(abs_pose_gt):
        raise ValueError(f"Can't build edge between node {idx_i} and node {idx_j}, there's only {len(abs_pose_gt)} nodes in the list!")
    
    abs_pose_i2w = abs_pose_gt[idx_i]
    abs_pose_j2w = abs_pose_gt[idx_j]
    rel_pose_i2j = torch.linalg.inv(abs_pose_j2w) @ abs_pose_i2w

    noise_rot = torch.Tensor(Rotation.from_euler('xyz', measurement_noise*5*(2*torch.randn(3)-1), degrees=True).as_matrix())
    noise_trans = measurement_noise * 5 * (2*torch.randn(3)-1)
    noise_mat = torch.eye(4)
    noise_mat[:3, :3] = noise_rot
    noise_mat[:3, 3] = noise_trans
    relative_pose_measure = noise_mat @ rel_pose_i2j

    rel_pose_dict_loop[f"pose_{idx_i}_{idx_j}"] = relative_pose_measure


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

    point_list, abs_pose_gt, abs_pose_noisy, rel_pose_dict_loop = create_data()
    add_loop_closure_constraint(abs_pose_gt, rel_pose_dict_loop, 0, len(point_list)-1, 0.2)

    print("Constructing a pose graph for Gaussian Splatting SLAM.")
    pose_graph = GaussianSLAMPoseGraph(requires_auto_grad=True)

    for idx in range(len(point_list)-1):

        abs_pose_i = abs_pose_noisy[idx]
        rotation_i = Rotation.from_matrix(abs_pose_i[:3, :3])
        quaternion_i = torch.Tensor(rotation_i.as_quat())
        xyz_quaternion_i = torch.cat((abs_pose_i[:3, 3], quaternion_j))
        vertex_i = th.SE3(xyz_quaternion_i)
            
        abs_pose_j = abs_pose_noisy[idx+1]
        rotation_j = Rotation.from_matrix(abs_pose_j[:3, :3])
        quaternion_j = torch.Tensor(rotation_j.as_quat())
        xyz_quaternion_j = torch.cat((abs_pose_j[:3, 3], quaternion_j))
        vertex_j = th.SE3(xyz_quaternion_j)

        rel_pose_ij = torch.linalg.inv(abs_pose_j) @ abs_pose_i
        pose_graph.add_odometry_edge(vertex_i, vertex_j, rel_pose_ij, point_list[idx])
    
    for key, value in rel_pose_dict_loop:
        # it will be a little bit complicated to get indices from key, so I just hard code here
        idx_i, idx_j = 0, 10

        abs_pose_i = abs_pose_noisy[idx]
        rotation_i = Rotation.from_matrix(abs_pose_i[:3, :3])
        quaternion_i = torch.Tensor(rotation_i.as_quat())
        xyz_quaternion_i = torch.cat((abs_pose_i[:3, 3], quaternion_j))
        vertex_i = th.SE3(xyz_quaternion_i)
            
        abs_pose_j = abs_pose_noisy[idx+1]
        rotation_j = Rotation.from_matrix(abs_pose_j[:3, :3])
        quaternion_j = torch.Tensor(rotation_j.as_quat())
        xyz_quaternion_j = torch.cat((abs_pose_j[:3, 3], quaternion_j))
        vertex_j = th.SE3(xyz_quaternion_j)

        pose_graph.add_loop_closure_edge(vertex_i, vertex_j, 1, value, point_list[idx_i], coefficient=1)

    info = pose_graph.optimize()




