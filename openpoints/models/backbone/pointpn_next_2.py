"""Official implementation of PointNext
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
https://arxiv.org/abs/2206.04670
Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem
"""
from typing import List, Type
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
import pdb


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

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

        # expand_p = 2 if last else 1
        # self.transfer = ConvBNReLU1D(out_channels * expand_p, out_channels, bias=False, activation=activation)
        
    def forward(self, xyz, re_xyz, x):
        B, _, npoint, nsample = re_xyz.shape
        # x = self.transfer(x.reshape(B, -1, npoint * nsample)).reshape(B, -1, npoint, nsample)
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
        # aggregation_features = x + position_embedding
        # aggregation_features *= position_embedding
        aggregation_features = x * position_embedding
        aggregation_features += position_embedding

        return aggregation_features


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
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]#+torch.mean(x, dim=-1, keepdim=False)
        elif reduction == 'mean':
            self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
        elif reduction == 'sum':
            self.pool = lambda x: torch.sum(x, dim=-1, keepdim=False)

        self.pos_embedding = PosPool_Block(channels[-1])

    def forward(self, px) -> torch.Tensor:
        # p: position, x: feature
        p, x = px
        # neighborhood_features
        dp, xj = self.grouper(p, p, x)
        _, xj = self.pos_embedding(p, dp, xj)
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
                 all_aggr=False,
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        # current blocks aggregates all spatial information.
        #self.all_aggr = not is_head and stride == 1
        self.all_aggr = all_aggr
        self.use_res = use_res and not self.all_aggr and not self.is_head

        mid_channel = out_channels // 4 if not self.is_head else out_channels
        # mid_channel = out_channels // 4 if stride > 1 else out_channels
        # mid_channel = out_channels // 2
        channels = [in_channels] + [mid_channel] * \
            (layers - 1) + [out_channels]
        # channels[0] = in_channels #+ 3 * (not is_head)
        if self.is_head or self.all_aggr:
        #if self.all_aggr:
            channels[0] = in_channels #+ 3 * (not is_head)
        else:
            channels[0] = 2 * in_channels #+ 3 * (not is_head)
        # if self.all_aggr:
        #     channels[-1] *= 2
        # if self.use_res:
        #     self.skipconv = create_convblock1d(
        #         in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
        #         -1] else nn.Identity()
        self.act = create_act(act_args)
        create_conv = create_convblock1d #if is_head else create_convblock2d
        convs = []
        if self.stride != 1 or self.is_head:
            for i in range(len(channels) - 1):
                convs.append(create_conv(channels[i], channels[i + 1],
                                        norm_args=norm_args,
                                        act_args=None if i == len(channels) - 2
                                        and (self.use_res) else act_args,
                                        **conv_args)
                            )
        else:
            channels[0] = channels[0] // 2
            for i in range(len(channels) - 1):
                convs.append(create_conv(channels[i], channels[i + 1],
                                        norm_args=norm_args,
                                        act_args=None if i == len(channels) - 2
                                        and (self.use_res) else act_args,
                                        **conv_args)
                            )
        # convs.append(create_conv(channels[0], channels[0],
        #                         norm_args=norm_args if not is_head else None,
        #                         act_args=act_args,
        #                         **conv_args)
        #             )
        # convs.append(create_conv(channels[0], channels[-1],
        #                         norm_args=norm_args if not is_head else None,
        #                         act_args=None if (self.use_res or is_head) else act_args,
        #                         **conv_args)
        #             )
        self.convs = nn.Sequential(*convs)
        if self.all_aggr:
        #if last:
            group_args.nsample = None
            group_args.radius = None
        if not self.is_head:
            self.grouper = create_grouper(group_args)
        #self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0] +torch.mean(x, dim=-1, keepdim=False)
        if sample_method.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif sample_method.lower() == 'random':
            self.sample_fn = random_sample

        # if stride <= 1 and self.all_aggr:
        #     last = True
        # else:
        #     last = False

        # if self.all_aggr:
        #     self.pos_embedding = PosPool_Block(channels[-1])
        # else:
        #     self.pos_embedding = PosPool_Block(channels[-1])
        if self.stride != 1:
            self.transfer = create_convblock1d(channels[-1], channels[-1],
                                            norm_args=norm_args,
                                            act_args=act_args,
                                            **conv_args
                        )
            self.pos_embedding = PosPool_Block(channels[-1])
        elif self.stride == 1 and not self.is_head:
            self.transfer = create_convblock1d(channels[-1], channels[-1]//2,
                                            norm_args=norm_args,
                                            act_args=act_args,
                                            **conv_args
                        )
            self.pos_embedding = PosPool_Block(channels[-1]//2)

    def forward(self, px):
        p, x = px
        if not self.all_aggr:
            idx = self.sample_fn(p, p.shape[1] // self.stride).long()
            new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        else:
            new_p = p
        if self.is_head:
            x = self.convs(x)
            # dp, xj = self.grouper(new_p, p, x, idx)
            # pdb.set_trace()
            # _, xj_new = self.pos_embedding(new_p, dp, xj)
            # x = self.convs(self.pool(xj_new))
            # x += F.adaptive_max_pool2d(xj_new, (1024, 1)).squeeze()
            # x = self.conv(torch.cat(x.unsqueeze(-1), xj_new), dim=3)
        else:
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
            if not self.all_aggr:
                dp, xj = self.grouper(new_p, p, x, idx) # # (B, 3, npoint, nsample) (B, width, npoint, nsample)
            else:
                dp, xj = self.grouper(new_p, p, x)
            B, _, npoint, nsample = xj.shape
            xj = self.act(xj)
            xj = self.transfer(xj.reshape(B, -1, npoint*nsample)).reshape(B, -1, npoint, nsample)
            _, xj_new = self.pos_embedding(new_p, dp, xj)
            #x = self.pool(self.convs(torch.cat((dp, xj_new), dim=1)))
            #x = self.pool(xj_new)
            x = self.convs(self.act(self.pool(xj_new)))
            # x = self.pool(self.convs(xj_new))
            # x = self.convs(self.pool(torch.cat((dp, xj_new), dim=1)))
            # if self.use_res:
            #     x = self.act(x + identity)
            p = new_p
            # if self.all_aggr:
            #     x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1) + x.mean(-1)
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
class PointPN_next(nn.Module):
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
                width *= 2 #2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            # if i == len(blocks)-1:
            #     last = True
            # else:
            #     last = False
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            all_aggr = True if i == len(blocks) - 1 else False
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1, all_aggr=all_aggr
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

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False, all_aggr=False):
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
                                     is_head=is_head, use_res=self.sa_use_res, all_aggr=all_aggr
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
        for i in range(0, len(self.encoder)-1):
            p0, f0 = self.encoder[i]([p0, f0])
        f0 = F.adaptive_max_pool1d(f0, 1).squeeze(dim=-1) + f0.mean(-1)
        return f0 # bs dim

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