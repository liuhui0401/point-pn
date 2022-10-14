# group layer: find neighbors for each point
# knn, knn_sparse, ball query

# gather layer, gather features by index
from typing import Tuple
import copy, logging
import torch
import torch.nn as nn
from torch.autograd import Function
from openpoints.cpp import pointnet2_cuda
import pdb

class KNN(nn.Module):
    def __init__(self, neighbors, transpose_mode=True):
        super(KNN, self).__init__()
        self.neighbors = neighbors

    @torch.no_grad()
    def forward(self, support, query):
        """
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]
        Returns:
            [int]: neighbor idx. [B, M, K]
        """
        dist = torch.cdist(support, query)
        k_dist = dist.topk(k=self.neighbors, dim=1, largest=False)
        return k_dist.values, k_dist.indices.transpose(1, 2).contiguous().int()

# dilated knn
class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    index: (B, npoint, nsample)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, randnum]
            else:
                edge_index = edge_index[:, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, ::self.dilation]
        return edge_index.contiguous()


class DilatedKNN(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DilatedKNN, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = KNN(k * self.dilation, transpose_mode=True)

    def forward(self, query):
        _, idx = self.knn(query, query)
        return self._dilated(idx)


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample, device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint, device=features.device)

        pointnet2_cuda.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):  # todo: understand this part. why needs this backward??
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample, device=xyz.device).zero_()
        pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 normalize_by_std=False,
                 normalize_by_allstd=False,
                 normalize_by_allstd2=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            radius (float): radius of ball
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2   # only nomalize by one method
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None#, center_idx: torch.Tensor=None
            ) -> Tuple[torch.Tensor]:
        """
        :param query_xyz: (B, npoint, 3) xyz coordinates of the features
        :param support_xyz: (B, N, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)

        if self.return_only_idx:
            return idx
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        #center_xyz = grouping_operation(xyz_trans, center_idx.unsqueeze(-1).int())
        if self.relative_xyz:
            grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
        if self.normalize_dp:
            grouped_xyz /= self.radius
        if features is not None:
            grouped_features = grouping_operation(features, idx) # (B, width, npoint, nsample)
            return grouped_xyz, grouped_features
            #center_features = grouping_operation(features, center_idx.unsqueeze(-1).int())
            # if self.normalize_dp:
            #     mean = center_features
            #     std = torch.std(grouped_features-mean)
            #     mean_xyz = center_xyz
            #     std_xyz = torch.std(grouped_xyz-mean_xyz)
            #     grouped_features = (grouped_features-mean) / (std+1e-5)
            #     #grouped_features = self.affine_alpha * grouped_features + self.affine_beta
            #     grouped_xyz = (grouped_xyz-mean_xyz) / (std_xyz+1e-5)
            #return torch.cat([grouped_xyz, center_xyz.repeat(1, 1, 1, grouped_xyz.shape[-1])], dim=-1), torch.cat([grouped_features, center_features.repeat(1, 1, 1, grouped_features.shape[-1])], dim=-1)
            # return grouped_xyz, torch.cat([grouped_features, center_features.repeat(1, 1, 1, grouped_features.shape[-1])], dim=1)
        else:
            return grouped_xyz, None


class GroupAll(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, new_xyz: torch.Tensor, xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            return grouped_xyz, grouped_features
        else:
            return grouped_xyz, None


class KNNGroup(nn.Module):
    def __init__(self, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.nsample = nsample
        self.knn = KNN(nsample, transpose_mode=True)
        self.relative_xyz = relative_xyz
        self.normalize_dp = normalize_dp
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None, center_idx: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        :param query_xyz: (B, N, 3) xyz coordinates of the features
        :param support_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        _, idx = self.knn(support_xyz, query_xyz)
        if self.return_only_idx:
            return idx
        idx = idx.int()
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        center_xyz = grouping_operation(xyz_trans, center_idx.unsqueeze(-1).int())
        if self.relative_xyz:
            grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
        if self.normalize_dp:
            grouped_xyz /= torch.abs(grouped_xyz).max(dim=-1, keepdim=True)[0]
        if features is not None:
            grouped_features = grouping_operation(features, idx) # (B, width, npoint, nsample)
            # return grouped_xyz, grouped_features
            center_features = grouping_operation(features, center_idx.unsqueeze(-1).int())
        #     if self.normalize_dp:
        #         mean = center_features
        #         std = torch.std(grouped_features-mean)
        #         mean_xyz = center_xyz
        #         std_xyz = torch.std(grouped_xyz-mean_xyz)
        #         grouped_features = (grouped_features-mean) / (std+1e-5)
        #         #grouped_features = self.affine_alpha * grouped_features + self.affine_beta
        #         grouped_xyz = (grouped_xyz-mean_xyz) / (std_xyz+1e-5)
            #return torch.cat([grouped_xyz, center_xyz.repeat(1, 1, 1, grouped_xyz.shape[-1])], dim=-1), torch.cat([grouped_features, center_features.repeat(1, 1, 1, grouped_features.shape[-1])], dim=-1)
            return grouped_xyz, torch.cat([grouped_features, center_features.repeat(1, 1, 1, grouped_features.shape[-1])], dim=1)
            return grouped_xyz, grouped_features
        else:
            return grouped_xyz, None


def create_grouper(group_args):
    group_args_copy = copy.deepcopy(group_args)
    method = group_args_copy.pop('NAME', 'ballquery')
    radius = group_args_copy.pop('radius', 0.1)
    nsample = group_args_copy.pop('nsample', 20)

    logging.info(group_args)
    if nsample is not None:
        if method == 'ballquery':
            grouper = QueryAndGroup(radius, nsample, **group_args_copy)
        elif method == 'knn':
            grouper = KNNGroup(nsample,  **group_args_copy)
    else:
        grouper = GroupAll()
    return grouper


if __name__ == "__main__":
    import time

    B, C, N = 2, 3, 40960
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, C], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    # --------------- debug downsampling
    from openpoints.models.layers.layer3d import RandomSample, random_sample, furthest_point_sample

    npoints = 10000
    # rs = RandomSample(num_to_sample=npoints)
    # query, _= rs(points)
    idx = random_sample(points, npoints)
    # torch gather is faster then operation gather. 
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    idx = furthest_point_sample(points, npoints).to(torch.int64)
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    # # --------------- debug KNN
    # knn = KNN(k=K, transpose_mode=True)
    # # knn to get the neighborhood

    # # compare time usage.
    # st = time.time()
    # for _ in range(100):
    #     _, knnidx = knn(points, query) # B G M
    #     idx_base = torch.arange(0, B, device=points.device).view(-1, 1, 1) * N
    #     idx = knnidx + idx_base
    #     idx = idx.view(-1)
    #     neighborhood = points.view(B * N, -1)[idx, :]
    #     neighborhood = neighborhood.view(B, npoints, K, 3).contiguous()
    #     # normalize
    #     neighborhood1 = neighborhood - query.unsqueeze(2)
    # print(time.time() - st)
    # # print(neighborhood1.shape, '\n', neighborhood1)

    # knngroup = KNNGroup(K)
    # # KNN Group is faster then above torch indexing when warpped in class.  
    # st = time.time()
    # for _ in range(100):
    #     neighborhood2 = knngroup(query, points)
    # print(time.time() - st)
    # # print(neighborhood2.shape, '\n', neighborhood2)
    # flag = torch.allclose(neighborhood1, neighborhood2.permute(0, 2, 3, 1))
    # print(flag)

    # ------------- debug ball query
    query_group = QueryAndGroup(0.1, K)

    st = time.time()
    for _ in range(100):
        # ball querying is 40 times faster then KNN 
        features = query_group(query, points)
    print(time.time() - st)
    print(features.shape)