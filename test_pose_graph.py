import torch
import theseus as th
from scipy.spatial.transform import Rotation

from src.utils.pose_graph_utils import dense_surface_alignment

points_1 = torch.empty(10, 3)
points_2 = torch.empty(10, 3)

Rot = Rotation.from_euler('xyz', [1.5, -2, 0.5], degrees=True)
rotation = torch.Tensor(Rot.as_matrix())
translation = torch.Tensor((1, 2, 1))
relative_pose = torch.eye(4)
relative_pose[:3, :3] = rotation
relative_pose[:3, 3] = translation
noise = 1e-3

for i in range(10):
    points_1[i, :] = (i+1)*torch.ones(3)
    points_2[i, :] = rotation @ points_1[i] + translation
    points_2[i, i%3] += noise
print(points_1)
print("")
print(points_2)
xyz_quaternion = torch.Tensor((0, 0, 0, 1, 0, 0, 0))
vertex_1 = th.SE3(xyz_quaternion)
vertex_2 = th.SE3(xyz_quaternion)

optim_vars = vertex_1, vertex_2
aux_vars = relative_pose, points_1
res = dense_surface_alignment(optim_vars=optim_vars, aux_vars=aux_vars)
print(f"Test result is {res*res/2}")


res_gt = 0
rel_pose_inv = torch.linalg.inv(relative_pose)
temp = rel_pose_inv - torch.eye(4)

for i in range(10):
    pt =torch.tensor((points_1[i,0], points_1[i,1], points_1[i,2] ,1))
    temp2 = temp @ pt
    res_gt += temp2 @ temp2

print(f"Test result ground truth is {res_gt}")
