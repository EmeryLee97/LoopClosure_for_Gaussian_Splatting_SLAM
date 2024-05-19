import torch
import theseus as th

from src.utils.pose_graph_utils import dense_surface_alignment

points_1 = torch.empty(10, 3)
points_2 = torch.empty(10, 3)

rotation = torch.Tensor(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
translation = torch.Tensor((1, 2, 1))
relative_pose = torch.eye(4)
relative_pose[:3, :3] = rotation
relative_pose[:3, 3] = translation
noise = 1e-3

for i in range(10):
    points_1[i] = (i+1)*torch.ones(3)
    points_2[i] = (i+1)*torch.ones(3)
    points_2[i, i%3] += noise

xyz_quaternion = torch.Tensor((0, 0, 0, 1, 0, 0, 0))
vertex_1 = th.SE3(xyz_quaternion)
vertex_2 = th.SE3(xyz_quaternion)
print(f"The rotation part is {vertex_1.rotation()}")
print(f"The translation part is {vertex_1.translation()}")
#vertex_1 = th.SE3._init_tensor()
#vertex_2 = th.SE3._init_tensor()
optim_vars = vertex_1, vertex_2
aux_vars = relative_pose, points_1
res = dense_surface_alignment(optim_vars=optim_vars, aux_vars=aux_vars)
print(f"Test result is {res}")
