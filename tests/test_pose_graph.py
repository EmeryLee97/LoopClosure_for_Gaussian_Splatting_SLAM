import torch
import theseus as th

from src.utils.pose_graph_utils import GaussianSLAMPoseGraph, dense_surface_alignment

points_1 = []
points_2 = []

rotation = torch.Tensor(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
translation = torch.Tensor((1, 2, 1))
relative_pose = torch.ones(4)
relative_pose[:3, :3] = rotation
relative_pose[:3, 3] = translation
noise = 1e-3

for i in range(10):
    pt_1 = i*torch.ones(3)
    points_1.append(pt_1)
    pt_2 = pt_1
    pt_2[i%3] = pt_1[i % 3] + noise
    points_2.append(pt_2)

vertex_1 = th.SE3._init_tensor()
vertex_2 = th.SE3._init_tensor()
optim_vars = vertex_1, vertex_2
aux_vars = relative_pose, points_1
res = dense_surface_alignment(optim_vars=optim_vars, aux_vars=aux_vars)
print("Test result is")
