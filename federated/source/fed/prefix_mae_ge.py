# prefix_fed_mae.py
import torch
import torch.nn as nn
from copy import deepcopy

def build_prefix_fed_mae(mae_model, args):
    model = FedMAEPrefix(
        mae_model, 
        mid_dim=args.fed_mid_dim,
        scale=args.fed_scale,
        local_parts=args.fed_local_parts,
        vanilla=args.fed_vanilla,
        depth=args.fed_prefix_depth,
        scene_dim=args.fed_scene_dim if hasattr(args, 'fed_scene_dim') else 64
    )
    return model

# Adapter, VanillaPrefix, AdapterAttention 类保持不变...
class Adapter(nn.Module):
    def __init__(self, dim, mid_dim=256, act_fn='tanh'):
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim
        self.down = nn.Linear(dim, mid_dim)
        if act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        else:
            raise NotImplementedError
        self.up = nn.Linear(mid_dim, dim * 2)

    def forward(self, x, scene_embed=None):
        kv = self.up(self.act_fn(self.down(x)))
        return kv
    
    def forward_with_scene(self, x, scene_embed):
        """为了接口统一"""
        return self.forward(x)
    
class SceneAwareAdapter(Adapter): # 场景调制
    def __init__(self, dim, mid_dim=256, scene_dim=64):
        super().__init__(dim, mid_dim)
        # 添加场景调制模块
        self.scene_modulation = nn.Sequential(
            nn.Linear(scene_dim, mid_dim),
            nn.Sigmoid()
        )
    def forward(self, x, scene_embed=None):
        """支持scene_embed参数"""
        x_down = self.down(x)
        
        if scene_embed is not None:
            modulation = self.scene_modulation(scene_embed)
            modulation = modulation.unsqueeze(1)
            x_down = x_down * (1 + 1 * modulation)
        
        # 调试信息（仅第一次）
            if not hasattr(self, 'scene_modulation_logged'):
                print(f"  场景特征形状: {scene_embed.shape}")
                print(f"  调制参数范围: [{modulation.min().item():.4f}, {modulation.max().item():.4f}]")
                self.scene_modulation_logged = True
        kv = self.up(self.act_fn(x_down))
        return kv
    
    def forward_with_scene(self, x, scene_embed):
        """显式调用场景版本"""
        return self.forward(x, scene_embed)
    

class VanillaPrefix(nn.Module):
    def __init__(self, dim, zero_init=False, mid_dim=256, act_fn='tanh'):
        super().__init__()
        self.dim = dim
        self.kv = nn.Linear(dim, dim * 2)
        if zero_init:
            torch.nn.init.zeros_(self.kv.weight)
            torch.nn.init.zeros_(self.kv.bias)

    def forward(self, x, scene_embed=None):
        kv = self.kv(x)
        # q = torch.zeros(size=(kv.shape[0], kv.shape[1], self.dim), device=kv.device)
        # return torch.concat((q, kv), dim=-1)
        return kv
    def forward_with_scene(self, x, scene_embed):
        """为了接口统一"""
        return self.forward(x)

class AdapterAttention(nn.Module):
    def __init__(self, base_attention, dim, mid_dim=256, p_scale=0.8, vanilla='-', scene_dim=64):
        super().__init__()
        # 保存原始注意力参数
        self.qkv = deepcopy(base_attention.qkv)
        self.num_heads = deepcopy(base_attention.num_heads)
        self.scale = deepcopy(base_attention.scale)
        self.attn_drop = deepcopy(base_attention.attn_drop)
        self.proj = deepcopy(base_attention.proj)
        self.proj_drop = deepcopy(base_attention.proj_drop)

        # 适配器生成Prefix
        if vanilla == 'zero':
            self.adapter = VanillaPrefix(dim, zero_init=True, mid_dim=mid_dim)
        elif vanilla == 'random':
            self.adapter = VanillaPrefix(dim, zero_init=False, mid_dim=mid_dim)
        else:
            self.adapter = SceneAwareAdapter(dim, mid_dim=mid_dim, scene_dim=scene_dim)
        self.p_scale = p_scale
        self.scene_dim = scene_dim  # 保存场景维度

    def forward(self, x, scene_embed=None):
        B, N, C = x.shape

        
        # 生成prefix（支持场景特征）
        if scene_embed is not None:
            # 检查adapter是否支持scene_embed参数
            if hasattr(self.adapter, 'forward_with_scene'):
                prefixes = self.adapter.forward_with_scene(x, scene_embed)
            elif hasattr(self.adapter, 'forward'):
                try:
                    # 尝试调用带scene_embed参数的forward
                    prefixes = self.adapter(x, scene_embed=scene_embed)
                except TypeError:
                    # 如果不支持，使用普通forward
                    prefixes = self.adapter(x)
            else:
                prefixes = self.adapter(x)
        else:
            prefixes = self.adapter(x)
        
        # 标准ViT的qkv处理 + Prefix
        original_qkv = self.qkv(x)  # 形状: (B, N, C*3)
        
        # 分离 q, k, v
        qkv = original_qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # 分离 prefixes 的 k 和 v 部分
        prefixes = prefixes.reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_prefix, v_prefix = prefixes.unbind(0)
        
        # 添加前缀到 k 和 v
        k = k + self.p_scale * k_prefix
        v = v + self.p_scale * v_prefix

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FedMAEPrefix(nn.Module):
    """联邦学习MAE:编码器本地化,解码器参与联邦通信"""
    
    def __init__(self, mae_model, mid_dim=256, scale=1.5,
                 local_parts=None, vanilla='-', depth=3, scene_dim=64):
        super(FedMAEPrefix, self).__init__()
        
        self.mid_dim = mid_dim
        self.scale = scale
        self.vanilla = vanilla
        self.depth = depth
        self.embed_dim = 128
        self.scene_dim = scene_dim  # 保存场景维度
        
        # 只有解码器的adapter保持本地化
        self.local_parts = local_parts if local_parts is not None else ['adapter']

        # 深拷贝MAE模型
        self.mae_model = deepcopy(mae_model)
    
        # 为编码器添加Prefix适配器
        print(">>> Calling encoder adapter patch")
        self._add_prefix_attention_to_encoder()
    
        # 设置参数冻结策略
        self._setup_parameter_groups()

    def get_module_by_path(self, model, path):
        """根据路径安全地获取模块"""
        parts = path.split('.')
        obj = model
        
        for part in parts:
            if part.isdigit() or part == '-1':
                # 处理数字索引
                if part == '-1':
                    idx = -1
                else:
                    idx = int(part)
                
                # 检查对象是否是ModuleList或类似的可索引对象
                if isinstance(obj, (nn.ModuleList, nn.Sequential)):
                    if abs(idx) < len(obj):
                        obj = obj[idx]
                    else:
                        raise IndexError(f"Index {idx} out of range for {type(obj).__name__} with length {len(obj)}")
                else:
                    raise AttributeError(f"Cannot index {type(obj).__name__} with {part}")
            else:
                # 处理属性名
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    # 尝试在子模块中查找
                    found = False
                    for name, module in obj.named_children():
                        if name == part:
                            obj = module
                            found = True
                            break
                    if not found:
                        raise AttributeError(f"Cannot find {part} in {type(obj).__name__}")
        
        return obj

    def _add_prefix_attention_to_encoder(self):
        import types
        print(">>> ENTER _add_prefix_attention_to_encoder")

        for i, block in enumerate(self.mae_model.blocks):
            base_attn = block.attn

            block.attn = AdapterAttention(
                base_attn,
                dim=self.mae_model.embed_dim,
                mid_dim=self.mid_dim,
                p_scale=self.scale,
                vanilla=self.vanilla,
                scene_dim=self.scene_dim
            )

            # 直接定义 forward
            def forward_with_scene(self, x, scene_embed=None):
                norm1 = self.norm1(x)
                attn_out = self.attn(norm1, scene_embed=scene_embed)
                attn_out = self.drop_path1(attn_out)
                x = x + attn_out

                norm2 = self.norm2(x)
                mlp_out = self.mlp(norm2)
                mlp_out = self.drop_path2(mlp_out)
                x = x + mlp_out
                return x

            # 绑定
            block.forward = types.MethodType(forward_with_scene, block)

            print(f"[Encoder] Patched block {i} forward() with scene support")




    def _setup_parameter_groups(self):
        """Encoder共享 + Encoder Adapter个性化 + Decoder完全本地"""

        # 1. 冻结全部
        for p in self.parameters():
            p.requires_grad = False

        # 2. Encoder 主干（参与联邦）
        for name, p in self.mae_model.named_parameters():
            if (
                'patch_embed' in name or
                'blocks' in name or
                'norm' in name or
                'cls_token' in name
            ) and 'decoder' not in name:
                p.requires_grad = True

        # 3. Decoder 全部本地
        for name, p in self.mae_model.named_parameters():
            if 'decoder' in name:
                p.requires_grad = True


    def get_global_params(self):
        global_params = {}

        for name, param in self.mae_model.patch_embed.named_parameters():
            global_params[f"mae_model.patch_embed.{name}"] = param.data.clone()

        for name, param in self.mae_model.blocks.named_parameters():
            global_params[f"mae_model.blocks.{name}"] = param.data.clone()

        for name, param in self.mae_model.norm.named_parameters():
            global_params[f"mae_model.norm.{name}"] = param.data.clone()

        if hasattr(self.mae_model, 'cls_token'):
            global_params["mae_model.cls_token"] = self.mae_model.cls_token.data.clone()

        print(f"Collected {len(global_params)} global encoder parameters")
        return global_params


    def get_local_params(self):
        local_params = {}

        for name, param in self.mae_model.decoder_embed.named_parameters():
            local_params[f"mae_model.decoder_embed.{name}"] = param.data.clone()

        for i, block in enumerate(self.mae_model.decoder_blocks):
            for name, param in block.named_parameters():
                local_params[f"mae_model.decoder_blocks.{i}.{name}"] = param.data.clone()

        for name, param in self.mae_model.decoder_norm.named_parameters():
            local_params[f"mae_model.decoder_norm.{name}"] = param.data.clone()

        for name, param in self.mae_model.decoder_pred.named_parameters():
            local_params[f"mae_model.decoder_pred.{name}"] = param.data.clone()

        print(f"Collected {len(local_params)} local decoder parameters")
        return local_params


    def load_global_params(self, global_params):
        """加载全局参数到解码器主体结构"""
        loaded_count = 0
        for key, global_param in global_params.items():
            try:
                # 注意：这里应该使用 self.mae_model 而不是 self
                path = key.replace("mae_model.", "")
                obj = self.get_module_by_path(self.mae_model, path)
                
                # 更新参数
                if hasattr(obj, 'data'):
                    obj.data.copy_(global_param)
                    loaded_count += 1
                else:
                    print(f"Warning: {key} does not have data attribute")
                    
            except (AttributeError, IndexError, KeyError) as e:
                print(f"Error loading global parameter {key}: {e}")
                continue
        
        print(f"Loaded {loaded_count} global parameters to decoder")

    def load_local_params(self, local_params):
        """加载本地参数到编码器和解码器适配器"""
        loaded_count = 0
        for key, local_param in local_params.items():
            try:
                # 注意：这里应该使用 self.mae_model 而不是 self
                path = key.replace("mae_model.", "")
                obj = self.get_module_by_path(self.mae_model, path)
                
                # 更新参数
                if hasattr(obj, 'data'):
                    obj.data.copy_(local_param)
                    loaded_count += 1
                else:
                    print(f"Warning: {key} does not have data attribute")
                    
            except (AttributeError, IndexError, KeyError) as e:
                print(f"Error loading local parameter {key}: {e}")
                continue
        
        print(f"Loaded {loaded_count} local parameters")

    def forward(self, imgs, targets=None, grad_mask=None, mask_ratio=0.75, scene_embed=None):
        """MAE前向传播"""
        # 存储scene_embed
        return self.mae_model(imgs, targets, grad_mask, mask_ratio, scene_embed=scene_embed)

    def get_trainable_params(self):
        """获取当前需要训练的参数（用于优化器）"""
        return [param for param in self.parameters() if param.requires_grad]