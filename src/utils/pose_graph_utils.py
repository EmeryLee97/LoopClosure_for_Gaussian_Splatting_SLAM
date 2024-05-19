import numpy as np
import torch
import cv2
import theseus as th
from scipy.spatial.transform import Rotation
from typing import List, Union, Tuple, Optional

from src.entities.gaussian_model import GaussianModel
from src.utils.utils import to_skew_symmetric


def select_gaussian_inliers() -> torch.Tensor:
    """
    Select inlier correspondences from two Gaussian clouds
    """
    pass


def compute_relative_pose(input1, input2):
    """ Caluculate the relative pose between two cameras using Gaussians inside the frustum
    Args:
    Returns:
    """
    pass


# TODO: the gaussian_means must keep the same, do I need to detatch them and .to(cpu)?
#       How to make sure l_ij is between [0, 1]? test this function
def dense_surface_alignment(
        optim_vars: Union[Tuple[th.SE3, th.SE3], Tuple[th.SE3, th.SE3, torch.Tensor]],
        aux_vars: Tuple[torch.Tensor, torch.Tensor]
        ) -> torch.Tensor:
        """
        Compute the dense surface alignment error between two vertices, can be used as the error
        input to instantiate a th.CostFunction variable

        Args:
            optim_vars: optimizaiton variables registered in cost function, should contain
                vertex_i, vertex_j: correction matrix for pose i, j, 
                l_ij (optional): line process coefficient

            aux_vars: auxiliary variables registered in cost function, should contain
                relative_pose: constraint between vertex_i and vertex_j
                gaussian_means: mean positions of the 3D Gaussians inside frustum i

        Returns:
            square root of global place recognition error
        """

        # determine whether the edge is odometry edge or loop closure edge
        tuple_size = len(optim_vars)
        if tuple_size == 2:
            vertex_i, vertex_j = optim_vars
        elif tuple_size == 3:
            vertex_i, vertex_j, l_ij = optim_vars
        else:
            raise ValueError(f"optim_vars tuple size is {tuple_size}, which can only be 2 or 3.")
        relative_pose, gaussian_means = aux_vars

        pose_i = vertex_i.to_matrix().squeeze() # (4, 4) tensor
        pose_j = vertex_j.to_matrix().squeeze()

        if pose_i.size() != torch.Size((4, 4)):
            raise ValueError(f"Expected tensor size {(4, 4)}, but got {pose_i.size()}.")
        
        if pose_j.size() != torch.Size((4, 4)):
            raise ValueError(f"Expected tensor size {(4, 4)}, but got {pose_j.size()}.")
        
        pose_mat: torch.Tensor = torch.inverse(relative_pose) @ torch.inverse(pose_j) @ pose_i
        rot_mat: np.ndarray = pose_mat[:3, :3].numpy()
        rot:Rotation = Rotation.from_matrix(rot_mat)
        axis_angle_rotation: np.ndarray = rot.as_rotvec()
        axis_angle: torch.Tensor = torch.Tensor(axis_angle_rotation)
        trans: torch.Tensor = pose_mat[:3, 3]

        xi = torch.cat((axis_angle, trans))
        print(xi)

        p_skew_symmetric = to_skew_symmetric(gaussian_means) # (n, 3， 3) tensor
        G_p = torch.cat((-p_skew_symmetric, torch.eye(3).unsqueeze(0).expand(gaussian_means.size()[0], -1, -1)), dim=-1) # (n, 3, 6) tensor
        G_p_square = G_p.transpose(1, 2) @ G_p # (n, 6, 6) tensor
        Lambda = torch.sum(G_p_square, dim=0) # (6, 6) tensor
        res = xi @ Lambda @ xi

        if tuple_size == 3:
            return l_ij.sqrt() * res.sqrt() * np.sqrt(2)
        else:
            return res.sqrt() * np.sqrt(2)


def line_process(
        optim_vars: torch.Tensor,
        aux_vars: Optional[Tuple] = None
    ) -> torch.Tensor:
    """
    Computes the line process error of a loop closrue edge, can be used as the error
    input to instantiate a th.CostFunction variable

    Args:
        optim_vars:
            l_ij: jointly optimized weight (l_ij ∈ [0, 1]) over the loop edges
            (note that the scaling factor mu is considered as cost_weight)
        aux_vars: nothing needs to be passed here

    Returns:
        square root of line process error
    """
    l_ij = optim_vars
    return l_ij.sqrt() - 1


"""
    Custom Cost Funtion inherits from 'th.CostFunction'
    contains 
        optimization variables inherits from 'th.Manifold'
        auxiliary variables
    implements
        error computation and Jacobian
    returns
        torch.tensor as its error
"""
# TODO: implement the jacobians() method
class AdjacentVerticeCost(th.CostFunction):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        vertex_i: th.SE3,
        vertex_j: th.SE3,
        relative_pose = torch.eye(4),
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)

        if not isinstance(vertex_i, th.SE3) or not isinstance(vertex_j, th.SE3):
            raise TypeError("Vertex should be a th.SE3.")
        if not vertex_i.dof() == vertex_j.dof():
            raise ValueError("Two vertices must have identical dof.")

        self._cost_weight = cost_weight
        self._vertex_i = vertex_i
        self._vertex_j = vertex_j
        self._relative_pose = relative_pose
        self.register_optim_vars(["vertex_i", "vertex_j"])
        # self.register_aux_vars([ ])

    def error(self, gaussian_means: torch.tensor) -> torch.Tensor:
        """
        Compute the edge constraint between two adjacent vertices
        Returns:
            global place recognition error
        """
        return dense_surface_alignment(self._vertex_i, self._vertex_j, self._relative_pose, gaussian_means)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # TODO: implementation
        """
        Returns:
            jacobians: a list of jacobians, The i-th jacobian has shape 
                       (batch_size, cf.dim(), i-th_optim_var.dof())
            error: self.error()
        """
        pass

    def dim(self) -> int:
        return self._vertex_i.dof

    def _copy_impl(self, new_name: Optional[str] = None) -> "AdjacentVerticeCost":
        return AdjacentVerticeCost(
            self._cost_weight.copy(), self._vertex_i.copy(), self._vertex_j.copy(), name=new_name
            )


class LoopClosureVerticeCost(th.CostFunction):
    def __init__():
        pass

    def error(self, gaussian_means: torch.tensor) -> torch.Tensor:
        pass

    def dim(self) -> int:
        return self._vertex_i.dof + 1
    
    def _copy_impl(self):
        pass


class GaussianSLAMPoseGraph:
    def __init__(
        self, 
        requires_auto_grad = True
    ):
        self._requires_auto_grad = requires_auto_grad

        self._objective = th.Objective()
        # hyperparameters
        self._optimizer = th.LevenbergMarquardt(
            objective=self._objective,
            max_iterations=100,
            step_size=1)
        self._layer = th.TheseusLayer(self._optimizer)

        """
        The input is provided as a dictionary, where the keys represent either 
            the optimization variables (which are paired with their initial values), 
            or the auxiliary variables (which are paired with their data).
        """
        # TODO: add variables to this dictionary everytime new edge is added
        #       create and add cost function to objective, depends on "auto_grad" 
        self._theseus_inputs = {} 

    def add_odometry_edge(
            self,
            vertex_i: th.SE3,
            vertex_j: th.SE3,
            relative_pose: torch.Tensor,
            gassian_means: torch.Tensor
        ):
        optim_vars = vertex_i, vertex_j
        aux_vars = relative_pose, gassian_means
        error_function = dense_surface_alignment(optim_vars=optim_vars, aux_vars=aux_vars)
        cost_weight = th.ScaleCostWeight(1)
        if self._requires_auto_grad:
            cost_function = th.AutoDiffCostFunction( # is the 3rd input right?
                optim_vars, error_function, 1, cost_weight, aux_vars
                )
            
        self._objective.add(cost_function)
        self._theseus_inputs.update({ # do I need to update aux vars?
            vertex_i.name: vertex_i.tensor, 
            vertex_j.name: vertex_j.tensor
            })

    def add_loop_closure_edge(
            self,
            vertex_i: th.SE3,
            vertex_j: th.SE3,
            l_ij: th.Variable, # specific type???
            relative_pose: torch.tensor,
            gaussian_means: torch.tensor,
            coefficient: float # hyperparameter, not the same as in the paper
        ):
        optim_vars = vertex_i, vertex_j, l_ij
        aux_vars = relative_pose, gaussian_means,

        error_function_1 = dense_surface_alignment(optim_vars=optim_vars, aux_vars=aux_vars)
        cost_weight_1 = th.ScaleCostWeight(1)

        error_function_2 = line_process(optim_vars=optim_vars, aux_vars=aux_vars)
        cost_weight_2 = th.ScaleCostWeight(coefficient)

        if self._requires_auto_grad:
            cost_function_1 = th.AutoDiffCostFunction(
                optim_vars, error_function_1, 1, cost_weight_1, aux_vars)
            cost_function_2 = th.AutoDiffCostFunction(
                optim_vars, error_function_2, 1, cost_weight_2, aux_vars)
            
        self._objective.add(cost_function_1)
        self._objective.add(cost_function_2)
        
        self._theseus_inputs.update({ # do I need to update aux vars?
            vertex_i.name: vertex_i.tensor, 
            vertex_j.name: vertex_j.tensor
            })
        
    def optimize(self):
        with torch.no_grad():
            _, info = self._layer.forward(
                self._theseus_inputs, 
                optimizer_kwargs={"track_best_solution":True, "verbose":True}
                )
        return info

    def to(self, *args, **kwargs):
        if self._poses is not None:
            for pose in self.poses:
                pose.to(*args, **kwargs)

        if self._edges is not None:
            for edge in self.edges:
                edge.to(*args, **kwargs)


