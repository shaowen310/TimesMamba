import math

import torch
import torch.nn as nn
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class QuadMamba(nn.Module):
    def __init__(
        self,
        d_model=96,
        expand=2,
        d_state=16,
        d_conv=3,
        n_scan_directions=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.n_scan_directions = n_scan_directions
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        # x, z; z for residual

        self.dwconv = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        x_proj = [
            nn.Linear(
                self.d_inner, (self.dt_rank + d_state * 2), bias=False, **factory_kwargs
            )
            for _ in range(n_scan_directions)
        ]
        # dim(dts, Bs, Cs) = dt_rank, d_state, d_state
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in x_proj], dim=0)
        )  # (K, dt_rank + d_state*2, d_inner)

        self.dt_projs = [
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(n_scan_directions)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=n_scan_directions, merge=True
        )  # (K * D, N)
        self.Ds = self.D_init(
            self.d_inner, copies=n_scan_directions, merge=True
        )  # (K * D)

        self.selective_scan = selective_scan_fn

        self.out_proj = nn.Identity()
        if self.d_inner != self.d_model:
            self.out_proj = nn.Linear(
                self.d_inner, self.d_model, bias=bias, **factory_kwargs
            )
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        """
        x: (B, DI, H, W)

        B: batch size
        DI: d_inner = d_model * expand
        H: height, aka. the number of variates
        W: width, aka. sequence length
        """
        B, DI, H, W = x.shape
        K = self.n_scan_directions
        L = H * W

        x = x.view(B, DI, L)  # b d_inner l

        # # compute consine similarity
        # x_avg = torch.avg_pool1d(x.detach(), kernel_size=L)
        # sim = torch.cosine_similarity(x.detach(), x_avg, dim=1)

        # # rearrange scan order
        # scan_order = torch.argsort(sim, dim=1)
        # scan_order_tiled = scan_order.unsqueeze(2).tile((1, 1, DI)).transpose(1, 2)
        # x = torch.gather(x, 2, scan_order_tiled)

        # scan direction, [width right, height down] and [height down, width right]
        x_wrhd_hdwr = x.unsqueeze(1)  # b 1 d_inner l

        if K == 4:
            x_wrhd_hdwr = torch.stack(
                [
                    x.view(B, -1, L),
                    torch.transpose(x, 2, 3).contiguous().view(B, DI, L),
                ],
                dim=1,
            ).view(
                B, 2, DI, L
            )  # (b, 2, d, l)
        # scan direction, [width left, height up] and [height up, width left]
        x_wlhu_huwl = torch.flip(x_wrhd_hdwr, dims=[-1])
        xs = torch.cat([x_wrhd_hdwr, x_wlhu_huwl], dim=1)  # (b, k, d_inner, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # xs: (b, k, d_inner, l)
        # self.x_proj_weight: (k, dt_rank + d_state*2, d_inner)
        # x_dbl: (b, k, dt_rank + d_state*2, l)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        # dts: (b, k, dt_rank, l)
        # Bs: (b, k, d_state, l)
        # Cs: (b, k, d_state, l)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        # self.dt_projs_weight: (k, d_inner, dt_rank)
        # dts: (b, k, d_inner, l)

        xs = xs.float().view(B, -1, L)  # (b, k*d_inner, l)
        dts = dts.float().contiguous().view(B, -1, L)  # (b, k*d_inner, l)
        As = -torch.exp(self.A_logs.float())  # (k*d_inner, d_state)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)
        Ds = self.Ds.float()  # (k*d_inner)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k*d_inner)

        y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)

        if K == 2:
            y[:, 1] = torch.flip(y[:, 1], dims=[-1]).view(B, -1, L)
        elif K == 4:
            y[:, 2:4] = torch.flip(y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            y[:, 1] = (
                torch.transpose(y[:, 1].view(B, -1, W, H), 2, 3)
                .contiguous()
                .view(B, -1, L)
            )
            y[:, 3] = (
                torch.transpose(y[:, 3].view(B, -1, W, H), 2, 3)
                .contiguous()
                .view(B, -1, L)
            )

        y = torch.sum(y, dim=1)  # b d_inner l

        # # restore the original order
        # inv_scan_order = torch.argsort(scan_order, dim=1)  # bs*pn nv
        # inv_scan_order_tiled = (
        #     inv_scan_order.unsqueeze(2).tile((1, 1, DI)).transpose(1, 2)
        # )
        # y = torch.gather(y, 2, inv_scan_order_tiled)  # bs*pn nv d_model

        y = y.view(B, -1, H, W)  # b d_inner h w

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: (B, H, W, C)

        B: batch size
        H: height, aka. the number of variates
        W: width, aka. sequence length
        C: channel == d_model
        """
        _, H, W, _ = x.size()

        xz = self.in_proj(x)  # (b, h, w, d_model) -> (b, h, w, d_inner * 2)
        # d_inner: d_model * expand
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d_inner)

        x = torch.permute(x, (0, 3, 1, 2)).contiguous()  # (b, d_inner, h, w)
        x = self.dwconv(x)[..., :H, :W]  # (b, d_inner, h, w)
        x = self.act(x)
        y = self.forward_core(x)  # (b, d_inner, h, w)
        assert y.dtype == torch.float32
        y = torch.permute(y, (0, 2, 3, 1)).contiguous()  # (b, h, w, d_inner)

        z = self.act(z)

        out = y * z
        out = self.out_proj(out)
        out = self.drop(out)
        return out


class gMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class MambaformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        # ==== ssm
        ssm_d_state: int = 16,
        ssm_expand: int = 1,
        ssm_conv: int = 3,
        ssm_directions: int = 2,
        ssm_drop_rate: float = 0.0,
        # ==== mlp
        mlp_ratio: float = 4.0,
        mlp_act_layer: nn.Module = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.mamba_branch = ssm_expand > 0

        if self.mamba_branch:
            self.norm = norm_layer(hidden_dim)
            self.mamba = QuadMamba(
                d_model=hidden_dim,
                expand=ssm_expand,
                d_state=ssm_d_state,
                d_conv=ssm_conv,
                n_scan_directions=ssm_directions,
                dropout=ssm_drop_rate,
                **kwargs,
            )

        self.mlp_branch = mlp_ratio > 0

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
            )

        if self.mamba_branch or self.mlp_branch:
            self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # input: (b, h, w, c)

        x = input

        if self.mamba_branch:
            x = x + self.drop_path(self.mamba(self.norm(input)))  # SSM

        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN

        return x


class Mambaformer(nn.Module):
    """A basic Mamba layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        depth,
        expand=2,
        ssm_drop_rate=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        ssm_d_state=16,
        ssm_conv=3,
        ssm_directions=2,
        mlp_ratio=0,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                MambaformerLayer(
                    dim,
                    ssm_expand=expand,
                    ssm_d_state=ssm_d_state,
                    ssm_conv=ssm_conv,
                    ssm_directions=ssm_directions,
                    ssm_drop_rate=ssm_drop_rate,
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        # x: (b, h, w, c), l: seq_len

        for blk in self.blocks:
            x = blk(x)  # (b, h, w, c)

        return x


class MambaForSeriesForecasting(nn.Module):
    def __init__(
        self,
        dims=[768],
        depths=[4],
        ssm_expand=2,
        ssm_d_state=16,
        ssm_conv=3,
        ssm_directions=2,
        ssm_drop_rate=0.1,
        mlp_ratio=4,
        mlp_drop_rate=0.1,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_layers = len(depths)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Mambaformer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                expand=ssm_expand,
                ssm_d_state=ssm_d_state,
                ssm_conv=ssm_conv,
                ssm_directions=ssm_directions,
                ssm_drop_rate=ssm_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
            )
            self.layers.append(layer)

        self.norm = norm_layer(dims[-1])

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (b, h, w, c)

        for layer in self.layers:
            x = layer(x)  # b h w c

        x = self.norm(x)  # b h w c

        return x
