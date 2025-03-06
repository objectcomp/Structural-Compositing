import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepBlock,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock
)
from ldm.util import exists

class CannyEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),  
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), 
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2), 
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        
        return self.net(x)

class HedEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),  
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), 
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2), 
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
    
        return self.net(x)

class DepthEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
    
        return self.net(x)


class ConditionGating(nn.Module):
    
    def __init__(self, num_streams=3, in_channels=128, gating_type='scalar'):
        super().__init__()
        self.gating_type = gating_type

        if gating_type == 'scalar':
            self.gates = nn.Parameter(torch.ones(num_streams), requires_grad=True)
        elif gating_type == 'conv':
            self.gates = nn.Conv2d(num_streams * in_channels, num_streams * in_channels, 
                                   kernel_size=1, groups=num_streams, bias=True)
        else:
            raise ValueError("Unsupported gating_type")

    def forward(self, features_list):

        if self.gating_type == 'scalar':
            out = 0
            for i, feat in enumerate(features_list):
                alpha_i = self.gates[i]  # scalar
                out = out + alpha_i * feat
            return out

        elif self.gating_type == 'conv':
            fused = torch.cat(features_list, dim=1)   
            gated = self.gates(fused)               
            # sum-split
            B, totalC, H, W = gated.shape
            step = totalC // len(features_list)
            out = 0
            idx = 0
            for _ in range(len(features_list)):
                out = out + gated[:, idx:idx+step, :, :]
                idx += step
            return out


class FDN(nn.Module):
    
    def __init__(self, norm_nc, cond_nc):
        super().__init__()
        # group norm on the main feature
        self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)
        # conv to produce gamma,beta from cond
        self.mlp = nn.Conv2d(cond_nc, 2 * norm_nc, kernel_size=3, padding=1)

    def forward(self, x, cond):
        
        # 1) group norm
        normalized = self.param_free_norm(x)
        # 2) project cond -> gamma,beta
        cond_out = self.mlp(cond)
        gamma, beta = torch.chunk(cond_out, 2, dim=1)
        # 3) apply scale+shift
        out = normalized * (1.0 + gamma) + beta
        return out

class InjectResBlock(nn.Module):
    """
    A residual block that merges main feature x [B, main_ch, H, W]
    with condition feature cond [B, cond_ch, H, W]
    using FDN-based injection. 
    The output has 'out_channels' (default = main_channels).
    """
    def __init__(self, main_channels=320, cond_channels=128, out_channels=None, dropout=0.0):
        super().__init__()
        if out_channels is None:
            out_channels = main_channels

        self.main_channels = main_channels
        self.cond_channels = cond_channels
        self.out_channels  = out_channels

        # FDN for first and second part of the residual block
        self.norm_in  = FDN(main_channels, cond_channels)
        self.norm_out = FDN(out_channels, cond_channels)

        # Convolutions in the ResBlock
        self.conv_in  = nn.Conv2d(main_channels, out_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # zero-init last conv for stable training, 
        # reminiscent of ControlNet / Uni-ControlNet style
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

        # For skip-connection if out_channels != main_channels
        if out_channels == main_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(main_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, cond):
        """
        x:    [B, main_ch, H, W]
        cond: [B, cond_ch,  H, W]
        return: [B, out_ch, H, W]
        """

        h = self.norm_in(x, cond)
        h = F.silu(h)
        h = self.conv_in(h)

       
        h = self.norm_out(h, cond)
        h = F.silu(h)
        h = self.conv_out(h)

        return self.skip(x) + h

class Comp_Control(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,  
            local_channels,  
            inject_channels,
            inject_layers,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 4),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, "Need context_dim if using spatial transformer"

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1
        if num_head_channels == -1:
            assert num_heads != -1

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels  
        self.inject_layers = inject_layers

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("Mismatch: len(num_res_blocks) != len(channel_mult)")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

       
        self.canny_encoder = CannyEncoder(out_channels=128)
        self.hed_encoder   = HedEncoder(out_channels=128)
        self.depth_encoder = DepthEncoder(out_channels=128)

        
        self.gating = ConditionGating(num_streams=3, in_channels=128, gating_type='conv')

        
        self.inject_block = InjectResBlock(main_channels=model_channels, cond_channels=128, out_channels=model_channels)

        
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ])
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels

                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disable_self_attentions[level] if exists(disable_self_attentions) else False,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                block = TimestepEmbedSequential(*layers)
                self.input_blocks.append(block)
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                down_block = ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels=out_ch,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=True,
                ) if resblock_updown else Downsample(
                    ch, conv_resample, dims=dims, out_channels=out_ch
                )
                self.input_blocks.append(TimestepEmbedSequential(down_block))
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        # Middle block
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, timesteps, context, local_conditions, **kwargs):
        
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        
        canny_map = local_conditions[:, 0:3, :, :]  
        hed_map   = local_conditions[:, 3:6, :, :]  
        depth_map = local_conditions[:, 6:9, :, :]  

        feat_canny = self.canny_encoder(canny_map)   
        feat_hed   = self.hed_encoder(hed_map)       
        feat_depth = self.depth_encoder(depth_map)    

        
        fused_features = self.gating([feat_canny, feat_hed, feat_depth])

        outs = []
        h = x.type(self.dtype)  

        for layer_idx, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            
            h = module(h, emb, context)

           
            if layer_idx in self.inject_layers:
              
                h = self.inject_block(h, fused_features)

            
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs



class ControlUNetModel(UNetModel):

    def forward(self, x, timesteps=None, context=None, local_control=None, **kwargs):
        
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)

            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)

            h = self.middle_block(h, emb, context)

        
        h += local_control.pop()  
        for module in self.output_blocks:
            
            h = torch.cat([h, hs.pop() + local_control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)     