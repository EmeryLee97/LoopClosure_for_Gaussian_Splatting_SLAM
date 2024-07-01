import numpy as np
import torch
import theseus as th
import torchlie.functional as lieF # use this instead of th.SE3
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from typing import Union, List, Tuple, Optional, cast

def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.detach().cpu().numpy()


def to_skew_symmetric(tensor: torch.Tensor):
    """
    Transform a (3, ) tensor to a (3, 3) tensor, or
    Transform a (num_pts, 3) tensor to a (num_pts, 3, 3) tensor, or
    Transform a (batch_size, num_pts, 3) tensor to a (batch_size, num_pts, 3, 3) tensor

    Args:
        tensor (torch.Tensor): 3d point cloud(s) that need(s) to be transformed to skew symmetric matrix

    Returns:
        skew_symmetric (torch.Tensor): transformed skew symmetric matrices
    """

    tensor_shape = tensor.shape
    if len(tensor_shape) > 3 or tensor_shape[-1] != 3:
        raise ValueError("Incorrect tensor dimension!")

    if len(tensor_shape) == 1:
        skew_symmetric = tensor.new_zeros((1, )+tensor_shape+(3, ))
    else:
        skew_symmetric = tensor.new_zeros(tensor_shape+(3, ))

    skew_symmetric[..., 0, 1] = -tensor[..., 2]
    skew_symmetric[..., 0, 2] = tensor[..., 1]
    skew_symmetric[..., 1, 0] = tensor[..., 2]
    skew_symmetric[..., 1, 2] = -tensor[..., 0]
    skew_symmetric[..., 2, 0] = -tensor[..., 1]
    skew_symmetric[..., 2, 1] = tensor[..., 0]

    return skew_symmetric

class GaussianSLAMEdge:
    def __init__(
        self,
        vertex_idx_i: int,
        vertex_idx_j: int,
        relative_pose: th.SE3,
        cost_weight: th.CostWeight
    ):
        self.vertex_idx_i = vertex_idx_i
        self.vertex_idx_j = vertex_idx_j
        self.relative_pose = relative_pose
        self.cost_weight = cost_weight


class GaussianSLAMPoseGraph:
    def __init__(
        self, 
        requires_auto_grad = True
    ):
        self._requires_auto_grad = requires_auto_grad
        self._objective = th.Objective()
        self._theseus_inputs = {} 

    def add_odometry_edge(
            self,
            vertex_i: th.SE3,
            vertex_j: th.SE3,
            edge: GaussianSLAMEdge,
            gaussian_means: torch.Tensor
        ):

        relative_pose, cost_weight = edge.relative_pose, edge.cost_weight

        gaussian_means_th = th.Variable(tensor=gaussian_means.unsqueeze(0), name=f"gaussian_means_odometry__{edge.vertex_idx_i}_{edge.vertex_idx_j}")

        optim_vars = vertex_i, vertex_j
        aux_vars = relative_pose, gaussian_means_th
        if self._requires_auto_grad:
            cost_function = th.AutoDiffCostFunction(
                optim_vars, GaussianSLAMPoseGraph.dense_surface_alignment, 1, cost_weight, aux_vars
            )
            self._objective.add(cost_function)
                
            self._theseus_inputs.update({
                vertex_i.name: vertex_i.tensor, 
                vertex_j.name: vertex_j.tensor
            })
        else:
            raise NotImplementedError()

    def add_loop_closure_edge(
            self,
            vertex_i: th.SE3,
            vertex_j: th.SE3,
            edge: GaussianSLAMEdge,
            gaussian_means: torch.tensor,
            match_num : int, # kapa
            tau: float=0.2, # fairly liberal distance threshold
        ):

        relative_pose = edge.relative_pose
        cost_weight_alignment = edge.cost_weight # for dense surface alignment
        cost_weight_mu = edge.cost_weight.scale.tensor.squeeze() * np.sqrt(match_num) * tau
        print(f"cost_weight_mu = {cost_weight_mu}")
        cost_weight_line_process = th.ScaleCostWeight(cost_weight_mu) # for line process

        l_ij = th.Vector(tensor=torch.ones(1, 1), name=f"line_process_{edge.vertex_idx_i}_{edge.vertex_idx_j}")

        gaussian_means_th = th.Variable(tensor=gaussian_means.unsqueeze(0), name=f"gaussian_means_odometry__{edge.vertex_idx_i}_{edge.vertex_idx_j}")

        optim_vars = vertex_i, vertex_j, l_ij
        aux_vars = relative_pose, gaussian_means_th

        if self._requires_auto_grad:
            cost_function = th.AutoDiffCostFunction(
                optim_vars, GaussianSLAMPoseGraph.dense_surface_alignment, 1, cost_weight_alignment, aux_vars)
            self._objective.add(cost_function)

            cost_function = th.AutoDiffCostFunction(
                    [l_ij,], GaussianSLAMPoseGraph.line_process, 1, cost_weight_line_process)
            self._objective.add(cost_function)
        
            self._theseus_inputs.update({
                vertex_i.name: vertex_i.tensor, 
                vertex_j.name: vertex_j.tensor,
                l_ij.name: l_ij.tensor
            })
        else:
            raise NotImplementedError()

    def optimize(self, max_iterations=1e3, step_size=0.01, damping=0.1, track_best_solution=True, verbose=False):
        optimizer = th.LevenbergMarquardt(
            objective=self._objective,
            max_iterations=max_iterations,
            step_size=step_size)
        
        layer = th.TheseusLayer(optimizer)

        with torch.no_grad():
            _, info = layer.forward(
                self._theseus_inputs, 
                optimizer_kwargs={"damping": damping, "track_best_solution":track_best_solution, "verbose":verbose}
                )
        return info

    @ staticmethod
    def match_gaussian_means(
        pts_1: torch.tensor,
        pts_2: torch.tensor,
        transformation: torch.tensor,
        epsilon:float=5e-2
    ) -> List[Tuple[int, int]]:
        """
        Select inlier correspondences from two Gaussian clouds, use kd-tree to speed up
    
        Args:
            pts_1, pts_2: mean positions of 3D Gaussians
            transformation: prior transformation matrix from one Gaussian cloud to the other
            epsilon: threshold for finding inlier correspondence
    
        Returns:
            a list contains tuples of matching indices
        """
        if transformation.size() != torch.Size([4, 4]):
            raise ValueError(f"The size of input transformation matrix must be (4, 4), but get {transformation.size()}")
    
        if pts_1.size(-1) != 1:
            pts_1 = pts_1.unsqueeze(-1)
    
        if isinstance(pts_1, th.Point3) or isinstance(pts_2, th.Point3):
                raise TypeError("To be matched points must be torch.Tensor")
    
        rotation = transformation[:3, :3]
        translation = transformation[:3, 3]
        pts_1_new = (rotation @ pts_1).squeeze() + translation
    
        pts_1_numpy = torch2np(pts_1_new)
        pts_2_numpy = torch2np(pts_2)
        pts2_kdtree = KDTree(pts_2_numpy)
    
        _, query_idx = pts2_kdtree.query(pts_1_numpy, distance_upper_bound=epsilon, workers=-1)
    
        data_size = pts_1.size()[0]
        res_list = []
        for i in range(data_size):
            if query_idx[i] != data_size:
                res_list.append((i, query_idx[i]))
    
        return res_list, len(res_list)

    @ staticmethod
    def dense_surface_alignment(
        optim_vars: Union[Tuple[th.SE3, th.SE3], Tuple[th.SE3, th.SE3, th.Vector]],
        aux_vars: Tuple[th.SE3, th.Variable]
    ) -> torch.Tensor:
        """
        Compute the dense surface alignment error between two vertices, can be used as the error
        function input to instantiate a th.CostFunction variable

        Args:
            optim_vars: optimizaiton variables registered in cost function, should contain
                pose_i, pose_j: correction matrix for pose i, j
                l_ij (optional): line process coefficient

            aux_vars: auxiliary variables registered in cost function, should contain
                relative_pose: constraint between vertex_i and vertex_j
                gaussian_means_i: mean positions of the 3D Gaussians inside camera frustum, 
                    represented in coordinate i and coordinate (those in coordinate j are not needed),
                    shape = (batch_size, num_pts, dim)

        Returns:
            square root of global place recognition error
        """
        # determine whether the edge is odometry edge or loop closure edge
        tuple_size = len(optim_vars)
        if tuple_size == 2:
            pose_i, pose_j = optim_vars
        elif tuple_size == 3:
            pose_i, pose_j, l_ij = optim_vars
        else:
            raise ValueError(f"optim_vars tuple size is {tuple_size}, which can only be 2 or 3.")
        relative_pose, gaussian_means = aux_vars

        pose_ij : th.SE3 = pose_j.inverse().compose(pose_i) # (batch_size, 3, 4)
        pose_residual : th.SE3 = relative_pose.inverse().compose(pose_ij) # (batch_size, 3, 4)
        trans_rot_vec :torch.Tensor = pose_residual.log_map().unsqueeze(1) # (batch_size, 1, 6)
        xi = torch.cat([trans_rot_vec[..., -3:], trans_rot_vec[..., :3]], dim=-1) # (batch_size, 1, 6)
        
        p_skew_symmetric = to_skew_symmetric(gaussian_means.tensor) # (batch_size, num_pts, 3, 3)
        G_p = torch.cat(( # (batch_size, num, 3, 6)
            -p_skew_symmetric, 
            torch.eye(3).reshape(1, 1, 3, 3).expand(gaussian_means.shape[0], gaussian_means.shape[-2], -1, -1)
            ), dim=-1)
        Lambda = torch.sum(G_p.transpose(-2, -1) @ G_p, axis=1) # (batch_size, 6, 6)
        res = (xi @ Lambda @ xi.transpose(-2, -1)).squeeze(1) # (batch_size, 1)
        
        if tuple_size == 3:
            return l_ij.tensor.sqrt() * res
        else:
            return res
        
    @ staticmethod
    def line_process(optim_vars: th.Vector, aux_vars=None) -> torch.Tensor:
        """
        Computes the line process error of a loop closrue edge, can be used as the error
        input to instantiate a th.CostFunction variable

        Args:
            optim_vars:
                l_ij: jointly optimized weight (l_ij ∈ [0, 1]) over the loop edges
                (note that the scaling factor mu is considered as cost_weight)

        Returns:
            square root of line process error
        """
        l_ij, = optim_vars
        return l_ij.tensor.sqrt() - 1
    

def create_data(
    num_pts: int = 1000, 
    num_poses: int = 10, 
    translation_noise: float = 0.05, 
    rotation_noise: float = 0.1, 
    weight = 2.0,
    batch_size: int = 1,
    #dtype = torch.float32 # will get error if changed to torch.float64, don't know why
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

    points_0 = th.Point3(2*torch.rand(num_pts, 3)-1, name="POINT_CLOUD__0") # initial points in world frame
    point_list = [points_0] # represented in different frames
    abs_pose_list_gt = [] # frame i to world frame
    abs_pose_list = [] # frame i to world frame (noisy)
    edge_list = []

    abs_pose_list_gt.append(th.SE3(
        tensor=torch.tile(torch.eye(3, 4), [1, 1, 1]),
        name="VERTEX_SE3_GT__0"
        ))
    
    abs_pose_list.append(th.SE3(
        tensor=torch.tile(torch.eye(3, 4), [1, 1, 1]),
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
        cost_weight = th.ScaleCostWeight(weight, name=f"EDGE_WEIGHT__{idx-1}_{idx}")

        # absolute pose of frame_{idx}
        absolute_pose_gt = cast(th.SE3, abs_pose_list_gt[-1].compose(relative_pose_gt.inverse()))
        absolute_pose_gt.name = f"VERTEX_SE3_GT__{idx}"

        absolute_pose = cast(th.SE3, abs_pose_list[-1].compose(relative_pose.inverse()))
        absolute_pose.name = f"VERTEX_SE3__{idx}"

        abs_pose_list_gt.append(absolute_pose_gt)
        abs_pose_list.append(absolute_pose)

        # construct odometry edge between vertex_{idx-1} and vertex_{idx}
        edge_list.append(GaussianSLAMEdge(idx-1, idx, relative_pose, cost_weight))

    return point_list, abs_pose_list_gt, abs_pose_list, edge_list


def add_loop_data(
        i: int, 
        j: int, 
        abs_pose_list_gt: List[th.SE3], 
        edge_list: List[GaussianSLAMEdge],
        weight: float = 2.0,
        measurement_noise:float = 0.01,
        batch_size: int = 1,
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
    rel_pose_ij = cast(th.SE3, relative_pose_noise.compose(rel_pose_ij_gt))
    rel_pose_ij.name = f"EDGE_SE3__{i}_{j}"

    cost_weight = th.ScaleCostWeight(weight, name=f"EDGE_WEIGHT__{i}_{j}")
    edge = GaussianSLAMEdge(i, j, rel_pose_ij, cost_weight)
    edge_list.append(edge)


if __name__ == "__main__":
    point_list, abs_pose_gt_list, abs_pose_list, edge_list = create_data()
    add_loop_data(0, 7, abs_pose_gt_list, edge_list)
    add_loop_data(1, 8, abs_pose_gt_list, edge_list)
    add_loop_data(2, 9, abs_pose_gt_list, edge_list)
    add_loop_data(0, 9, abs_pose_gt_list, edge_list)
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
            pose_graph.add_odometry_edge(vertex_i, vertex_j, edge, point_list[idx].tensor)
        else:
            print(f"adding edge {idx} to pose graph, current edge is an loop edge.")
            inlier_idx, num_matches = GaussianSLAMPoseGraph.match_gaussian_means(
                point_list[vertex_idx_i].tensor, point_list[vertex_idx_j].tensor, edge.relative_pose.to_matrix().squeeze(), epsilon=5e-2)
            inlier_idx_i = [idx_inlier[0] for idx_inlier in inlier_idx]
            pose_graph.add_loop_closure_edge(vertex_i, vertex_j, edge, point_list[vertex_idx_i].tensor[inlier_idx_i, :], num_matches, tau=0.2)

    #print(pose_graph._objective.error().shape)
    info = pose_graph.optimize(max_iterations=1e4, step_size=0.01, damping=0.1, verbose=False)
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

