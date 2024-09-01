""" This module includes the NetVLAD class and the LoopClosureDetector class, which are 
    responsible for extracting NetVLAD features and find loop frames
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import faiss # pip install faiss-gpu
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Union, Tuple
from src.utils.utils import np2torch, torch2np

"""
We thank Nanne https://github.com/Nanne/pytorch-NetVlad for the original design of the NetVLAD
class which in itself was based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py,
and contributers of Patch-NetVLAD https://github.com/QVPR/Patch-NetVLAD/blob/main/patchnetvlad/models/netvlad.py
for the modification.
"""

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, normalize_input=True, vladv2=False, use_faiss=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.use_faiss = use_faiss

    def init_params(self, clsts: np.ndarray, traindescs: np.ndarray):
        if not self.vladv2:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(axis=0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            if not self.use_faiss:
                knn = NearestNeighbors(n_jobs=-1)
                knn.fit(traindescs)
                del traindescs
                ds_sq = np.square(knn.kneighbors(clsts, 2)[1])
                del knn
            else:
                index = faiss.IndexFlatL2(traindescs.shape[1])
                # noinspection PyArgumentList
                index.add(traindescs)
                del traindescs
                # noinspection PyArgumentList
                ds_sq = np.square(index.search(clsts, 2)[1])
                del index
                
            self.alpha = (-np.log(0.01) / np.mean(ds_sq[:,1] - ds_sq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, ds_sq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2] # N: batch_size, C: channels=512 for vgg16 encoder

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            # residual: (N, num_clusters, C, H * W)
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C+1, :].unsqueeze(2)
            vlad[:, C:C+1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class LoopClosureDetector:

    def __init__(self, config: Dict, device='cuda'):
        self.ckpt_path = config["netvlad_checkpoint_path"]
        self.encoder_name = config["encoder_name"]
        self.k_neighbours = config["k_neighbours"]
        self.index_faiss = None
        self.device = device

        if self.encoder_name == 'vgg16':
            vgg16_model = models.vgg16()
            # capture only feature part and remove last relu layer and maxpool layer
            layers = list(vgg16_model.features.children())[:-2]
            encoder = nn.Sequential(*layers)
        else:
            raise NotImplementedError("Currently only support vgg16 as the encoder!")
        
        pool = NetVLAD(num_clusters=64, dim=512)
        self.model = nn.Module()
        self.model.add_module('encoder', encoder)
        self.model.add_module('pool', pool)
        self.model.to(self.device)
        checkpoint = torch.load(self.ckpt_path, map_location=torch.device(self.device), weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def get_netvlad_feature(self, image: np.ndarray) -> np.ndarray:
        """ calculate the netvlad feature of a given rgb image """
        if isinstance(image, np.ndarray):
            image = np2torch(image).permute(2, 0, 1).to(self.device)
        encoder_feature = self.model.encoder(image.unsqueeze(0).to(self.device))
        netvlad_feature = self.model.pool(encoder_feature)
        return torch2np(netvlad_feature)
        
    def add_to_index(self, image=None, netvlad_feature=None) -> None:
        """ Calculate the netvlad feature of an input image and add it to faiss index """
        if netvlad_feature is None:
            netvlad_feature = self.get_netvlad_feature(image)
        if self.index_faiss is None:
            self.netvlad_dim = netvlad_feature.shape[-1]
            if self.device == 'cuda': # use faiss-gpu
                res = faiss.StandardGpuResources()
                flat_config = faiss.GpuIndexFlatConfig()
                flat_config.device = 0
                self.index_faiss = faiss.GpuIndexFlatIP(res, self.netvlad_dim, flat_config)
            else: # use faiss-cpu
                self.index_faiss = faiss.IndexFlatIP(self.netvlad_dim)
                netvlad_feature = torch2np(netvlad_feature)
        self.index_faiss.add(netvlad_feature)
    
    def detect_knn(self, query_image=None, netvlad_feature=None, add_to_index=False, filter_threshold=None) \
        -> Tuple[Union[np.ndarray, torch.Tensor], List]:
        """ Transform the query image into a netvlad vector, search for k nearest neighbours of that feature 
            in faiss index, optionally remove all detected neighbours whose scores are lower than a given threshold.
        """
        if netvlad_feature is None:
            netvlad_feature = self.get_netvlad_feature(query_image)
        if self.get_index_length() < self.k_neighbours:
            k_neighbours = self.get_index_length()
        else:
            k_neighbours = self.k_neighbours

        score_list, idx_list = self.index_faiss.search(netvlad_feature, k_neighbours)
        score_list = score_list.squeeze()
        idx_list = idx_list.squeeze()

        # only keep neighbours that is closer than a threshold
        if filter_threshold is not None:
            filter_mask = score_list >= filter_threshold
            idx_list = idx_list[filter_mask]
        idx_list = idx_list.tolist()

        if self.get_index_length()-1 in idx_list:
            idx_list.remove(self.get_index_length()-1)

        if add_to_index:
            self.add_to_index(netvlad_feature=netvlad_feature)
        return score_list, idx_list
    
    def get_min_score(self, query_image=None, netvlad_feature=None) -> float:
        """ compute the netvlad feature of the input image, compare it to all elements inside the index and return the minimum score """
        if netvlad_feature is None:
            netvlad_feature = self.get_netvlad_feature(query_image)
        score_list, _ = self.index_faiss.search(netvlad_feature, self.get_index_length())
        score_list = min(score_list.squeeze().tolist())
        return score_list

    def reset(self) -> None:
        self.index_faiss.reset()

    def get_index_length(self) -> int:
        """ check the number of features stored in faiss index """
        return self.index_faiss.ntotal

    


