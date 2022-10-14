"""Official implementation of PointNext
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
https://arxiv.org/abs/2206.04670
Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem
"""
from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
import pdb


class PosPool_Block(nn.Module):
    def __init__(self, out_channels):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosPool_Block, self).__init__()
        self.out_channels = out_channels

        self.pos_layer = PosPool_Layer(out_channels)
        
    def forward(self, xyz, re_xyz, x):
        x = self.pos_layer(re_xyz, x) # B, nsample, npoints, embed_dim
        return xyz, x

class PosPool_Layer(nn.Module):
    def __init__(self, out_channels):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosPool_Layer, self).__init__()
        self.out_channels = out_channels
        

    def forward(self, re_xyz, x):
        B, _, npoint, nsample= re_xyz.shape
        feat_dim = self.out_channels // 6
        wave_length = 500
        alpha = 50
        feat_range = torch.arange(feat_dim, dtype=torch.float32).to(re_xyz.device)  # (feat_dim, )
        dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
        position_mat = torch.unsqueeze(alpha * re_xyz, -1)  # (B, 3, npoint, nsample, 1)
        div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, nsample, feat_dim)
        sin_mat = torch.sin(div_mat)  # (B, 3, npoint, nsample, feat_dim)
        cos_mat = torch.cos(div_mat)  # (B, 3, npoint, nsample, feat_dim)
        position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, nsample, 2*feat_dim)
        position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
        position_embedding = position_embedding.view(B, self.out_channels, npoint, nsample)  # (B, C, npoint, nsample)
        # aggregation_features = x * position_embedding  # (B, C, npoint, nsample)
        aggregation_features = x + position_embedding
        aggregation_features *= position_embedding
        # aggregation_features = x * position_embedding
        # aggregation_features += position_embedding

        return aggregation_features


class InitPosPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitPosPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(inplace=True))

    def forward(self, xyz):
        B, _, npoint = xyz.shape
        
        feat_dim = self.out_channels // (self.in_channels * 2)
        wave_length = 500
        alpha = 100
        
        feat_range = torch.arange(feat_dim, dtype=torch.float32).to(xyz.device)  # (feat_dim, )
        dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
        
        position_mat = torch.unsqueeze(alpha * xyz, -1)  # (B, 3, npoint, 1)
        div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, feat_dim)
        sin_mat = torch.sin(div_mat)  # (B, 3, npoint, feat_dim)
        cos_mat = torch.cos(div_mat)  # (B, 3, npoint, feat_dim)
        
        # position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, 2*feat_dim)

        # position_embedding = torch.zeros(B, 3, npoint, 2 * feat_dim).to(xyz.device)
        # position_embedding[:, :, :, 0::2] = sin_mat
        # position_embedding[:, :, :, 1::2] = cos_mat
        position_embedding = torch.stack([sin_mat, cos_mat], dim=4).flatten(3)


        position_embedding = position_embedding.permute(0, 1, 3, 2).contiguous()
        # position_embedding = position_embedding.permute(0, 3, 1, 2).contiguous()

        position_embedding = position_embedding.view(B, self.out_channels, npoint)  # (B, C, npoint)
        
        # position_embedding = position_embedding.view(B, self.out_channels // 3, 3, npoint) * xyz.unsqueeze(1)
        # position_embedding = position_embedding.reshape(B, self.out_channels, npoint)
        
        # position_embedding = self.out_transform(position_embedding)
        
        return position_embedding


class LocalAggregation(nn.Module):
    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(
                f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs = []
        for i in range(len(channels) - 1):  # #layers in each blocks
            convs.append(create_convblock2d(channels[i], channels[i + 1],
                                            norm_args=norm_args,
                                            act_args=None if i == (
                len(channels) - 2) and not last_act else act_args,
                **conv_args)
            )
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)

        reduction = 'mean' if reduction.lower() == 'avg' else reduction.lower()
        self.reduction = reduction
        assert reduction in ['sum', 'max', 'mean']
        if reduction == 'max':
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        elif reduction == 'mean':
            self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
        elif reduction == 'sum':
            self.pool = lambda x: torch.sum(x, dim=-1, keepdim=False)

    def forward(self, px) -> torch.Tensor:
        # p: position, x: feature
        p, x = px
        # neighborhood_features
        dp, xj = self.grouper(p, p, x)
        x = torch.cat((dp, xj), dim=1)
        x = self.convs(x)
        x = self.pool(x)
        """ DEBUG neighbor numbers. 
        if x.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return x


class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """
    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sample_method='fps',
                 use_res=False,
                 is_head=False,
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        # current blocks aggregates all spatial information.
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
            (layers - 1) + [out_channels]
        channels[0] = in_channels + 3 * (not is_head)

        # if self.use_res:
        #     self.skipconv = create_convblock1d(
        #         in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
        #         -1] else nn.Identity()
        #     self.act = create_act(act_args)
        # create_conv = create_convblock1d if is_head else create_convblock2d
        # convs = []
        # for i in range(len(channels) - 1):
        #     convs.append(create_conv(channels[i], channels[i + 1],
        #                              norm_args=norm_args if not is_head else None,
        #                              act_args=None if i == len(channels) - 2
        #                              and (self.use_res or is_head) else act_args,
        #                              **conv_args)
        #                  )
        # self.convs = nn.Sequential(*convs)
        if is_head:
            self.embedding = InitPosPool(channels[0], channels[-1])
        if not is_head:
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]+torch.mean(x, dim=-1, keepdim=False)
            if sample_method.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sample_method.lower() == 'random':
                self.sample_fn = random_sample
            self.pos_embedding = PosPool_Block(channels[-1])

    def forward(self, px):
        p, x = px
        if self.is_head:
            # x = self.convs(x)  # (n, c)
            x = self.embedding(x)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            """ DEBUG neighbor numbers. 
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            # if self.use_res:
            #     identity = torch.gather(
            #         x, -1, idx.unsqueeze(1).expand(-1, x.shape[1], -1))
            #     identity = self.skipconv(identity)
            # dp, xj = self.grouper(new_p, p, x)
            if not self.all_aggr:
                dp, xj = self.grouper(new_p, p, x, idx)
            else:
                dp, xj = self.grouper(new_p, p, x)
            _, xj = self.pos_embedding(new_p, dp, xj)
            x = self.pool(xj)

            # x = self.pool(self.convs(torch.cat((dp, xj), dim=1)))
            # if self.use_res:
            #     x = self.act(x + identity)
            p = new_p
        return p, x


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, px1, px2=None):
        # pxb1 is with the same size of upsampled points
        if px2 is None:
            _, x = px1  # (B, N, 3), (B, C, N)
            x_global = self.pool(x)
            x = torch.cat(
                (x, self.linear2(x_global).unsqueeze(-1).expand(-1, -1, x.shape[-1])), dim=1)
            x = self.linear1(x)
        else:
            p1, x1 = px1
            p2, x2 = px2
            x = self.convs(
                torch.cat((x1, three_interpolation(p1, p2, x2)), dim=1))
        return x


class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = in_channels * expansion
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, px):
        p, x = px
        identity = x
        x = self.convs([p, x])
        x = self.pwconv(x)
        if x.shape[-1] == identity.shape[-1] and self.use_res:
            x += identity
        x = self.act(x)
        return [p, x]


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = in_channels * expansion
        self.convs = LocalAggregation([in_channels, in_channels, mid_channels, in_channels],
                                      norm_args=norm_args, act_args=None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        self.act = create_act(act_args)

    def forward(self, px):
        p, x = px
        identity = x
        x = self.convs([p, x])
        if x.shape[-1] == identity.shape[-1] and self.use_res:
            x += identity
        x = self.act(x)
        return [p, x]


@MODELS.register_module()
class PointNN_next(nn.Module):
    r"""The Encoder for PointNext 
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 norm_args: dict = {'norm': 'bn'},
                 act_args: dict = {'act': 'relu'},
                 expansion: int = 4,
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = norm_args
        self.act_args = act_args
        self.conv_args = kwargs.get('conv_args', None)
        self.sample_method = kwargs.get('sample_method', 'fps')
        self.expansion = expansion
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sample_method=self.sample_method,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0['x']
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
        return f0.squeeze(-1)

    def forward_all_features(self, p0, x0=None):
        if hasattr(p0, 'keys'):
            p0, x0 = p0['pos'], p0['x']
        if x0 is None:
            x0 = p0.clone().transpose(1, 2).contiguous()
        p, x = [p0], [x0]
        for i in range(0, len(self.encoder)):
            _p, _x = self.encoder[i]([p[-1], x[-1]])
            p.append(_p)
            x.append(_x)
        return p, x

    def forward(self, p0, x0=None):
        self.forward_all_features(p0, x0)