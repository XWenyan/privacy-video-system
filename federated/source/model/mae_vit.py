import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Block
from timm.models.layers import DropPath
import numpy as np
from functools import partial

class PatchEmbed(nn.Module):
    """2D图像到patch嵌入的转换模块
    将输入图像分割为固定大小的patch,并通过卷积将每个patch映射到嵌入向量
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size) # 将patch大小转换为二维元组（h, w）
        
        # 用卷积层实现patch投影:输入通道数->嵌入维度，卷积核大小=patch大小，步长=patch大小
        # 每个卷积操作对应一个patch的特征提取
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=self.patch_size, 
                             stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() # 可选的归一化层

    def forward(self, x):
        B, C, H, W = x.shape # B:批次大小, C:通道数, H:高度, W:宽度
        # 确保尺寸能被patch_size整除
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
            f"Input image size ({H}*{W}) is not divisible by patch size {self.patch_size}"
        
        x = self.proj(x)  # 卷积得到patch特征:形状为(B, embed_dim, H/patch_h, W/patch_w)
        x = x.flatten(2).transpose(1, 2)  # 展平空间维度并转置:(B, embed_dim, N_patches) -> (B, N_patches, embed_dim)
        x = self.norm(x) # 归一化
        return x

def to_2tuple(x):
    """将输入转换为二维元组（处理单值或已为元组/列表的情况）
    例如:输入16 -> (16,16);输入(16,24) -> (16,24)
    """
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return (x[0], x[0])
        return x
    return (x, x)

class MaskedAutoencoderViT(nn.Module):
    """基于Vision Transformer的掩码自编码器(MAE)
    核心功能:通过掩码部分图像patch,让模型学习重建被掩码的内容，从而学习图像的有效特征
    扩展功能:支持异常分数计算，可能用于异常检测任务
    """
    def __init__(self, patch_size=16, in_chans=9, out_chans=4,
                 embed_dim=1024, depth=24, num_heads=16,  # 编码器参数
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, # 解码器参数
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_only_masked_tokens_ab=False, abnormal_score_func='L1', 
                 masking_method="random_masking", grad_weighted_loss=True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim # 编码器嵌入维度
        self.img_size = kwargs.get('img_size', (320, 640))
        
        # Abnormal specifics
        self.use_only_masked_tokens_ab = use_only_masked_tokens_ab # 是否仅用被掩码的patch计算异常分数
        self.abnormal_score_func = abnormal_score_func[0] if abnormal_score_func else 'L' # 是否使用梯度加权损失
        
        self.masking = getattr(self, masking_method) # 选择随机掩码
        self.grad_weighted_loss = grad_weighted_loss # 是否使用梯度加权损失（增强异常区域权重）
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.norm_pix_loss = norm_pix_loss # 是否对像素进行归一化后计算损失

        # 计算patch网格尺寸
        self.grid_h = self.img_size[0] // patch_size  # 320/16 = 20
        self.grid_w = self.img_size[1] // patch_size  # 640/16 = 40
        self.num_patches = self.grid_h * self.grid_w  # 20*40 = 800

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim, 
            norm_layer=norm_layer
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 编码器位置编码（固定，不参与训练）:为每个patch添加空间位置信息
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False
        )
        
        # 编码器Transformer块 堆叠多个Transformer块提取特征
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim) # 归一化

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True) # 编码器输出映射到解码器维度      
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) # 掩码token（用于填充被掩码的patch）
        
        # 解码器位置编码（固定，不参与训练）
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        # 解码器Transformer块 用于重建被掩码的patch
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim) # 归一化
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True) # 解码器预测层:将解码器输出映射为patch的像素值（patch_size^2 * out_chans:每个patch的所有像素）
        self.initialize_weights()

    def initialize_weights(self):
        """初始化模型权重（位置编码、patch嵌入、线性层等）"""
        encoder_pos_embed = get_2d_sincos_pos_embed(
            self.encoder_pos_embed.shape[-1], 
            self.grid_h, self.grid_w, cls_token=True
        )
        self.encoder_pos_embed.data.copy_(torch.from_numpy(encoder_pos_embed).float().unsqueeze(0))
        # 初始化解码器位置编码
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            self.grid_h, self.grid_w, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # 初始化patch嵌入
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # 初始化线性层和层归一化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """辅助初始化函数:对不同类型的层使用特定初始化方式"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # 偏置初始化为0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0) # 层归一化权重初始化为1
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """将图像转换为patch序列（用于与解码器输出对比计算损失）
        输入:imgs (B, out_chans, H, W)
        输出:(B, N_patches, patch_size^2 * out_chans) 每个patch展平为向量
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.out_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.out_chans))
        return x

    def unpatchify(self, x):
        """将patch序列还原为图像（用于可视化重建结果）
        输入:x (B, N_patches, patch_size^2 * out_chans)
        输出:(B, out_chans, H, W)
        """
        p = self.patch_size
        h = self.grid_h
        w = self.grid_w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio, grad_mask):
        """随机掩码patch:按mask_ratio比例掩码输入patch序列
        输入:x (B, N_patches, D) patch嵌入;mask_ratio 掩码比例
        输出:
            x_masked:保留的patch(未被掩码)
            mask:掩码标记(1表示被掩码,0表示保留)
            ids_restore:用于恢复原始patch顺序的索引
        """
        N, L, D = x.shape # N=批次大小, L=总patch数, D=嵌入维度
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device) # 生成随机噪声用于排序
        ids_shuffle = torch.argsort(noise, dim=1) # 按噪声排序得到打乱的索引
        ids_restore = torch.argsort(ids_shuffle, dim=1) # 恢复原始顺序的索引

        ids_keep = ids_shuffle[:, :len_keep] # 保留的patch索引
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # 提取保留的patch（按打乱后的顺序）

        # 生成掩码 1表示被掩码
        mask = torch.ones([N, L], device=x.device) 
        mask[:, :len_keep] = 0 # 保留的patch对应位置为0
        mask = torch.gather(mask, dim=1, index=ids_restore) # 恢复掩码的原始顺序

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, grad_mask, scene_embed=None):
        '''编码器前向传播'''
        # patch嵌入
        x = self.patch_embed(x)
        B, L, D = x.shape
        
        # 掩码操作
        x, mask, ids_restore = self.masking(x, mask_ratio, grad_mask)
        
        # 添加cls token和位置编码
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.encoder_pos_embed[:, :x.shape[1], :]

        # 通过Transformer块提取特征
        for blk in self.blocks:
            # 确保只传递位置参数，不传递关键字参数
            x = blk(x, scene_embed=scene_embed)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, scene_embed=None):
        '''解码器前向传播'''
        # 嵌入到解码器维度
        x = self.decoder_embed(x)
        
        # 计算需要添加的掩码token数量（原始patch数 - 保留的patch数）
        B = x.shape[0]
        num_original_patches = ids_restore.shape[1]  # 原始patch数量
        num_current_patches = x.shape[1] - 1  # 当前patch数量（减去cls_token）
        num_mask_tokens = num_original_patches - num_current_patches
        
        mask_tokens = self.mask_token.repeat(B, num_mask_tokens, 1) # 生成掩码token
        
        # 分离cls token和patch tokens
        cls_token = x[:, :1, :]  # cls token
        patch_tokens = x[:, 1:, :]  # 可见的patch tokens
        
        # 将可见patch和mask tokens拼接
        x_ = torch.cat([patch_tokens, mask_tokens], dim=1)   
        # 恢复原始顺序
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))     
        # 重新添加cls token
        x = torch.cat([cls_token, x_], dim=1)        
        # 添加解码器位置编码
        x = x + self.decoder_pos_embed
        # 应用Transformer块
        for blk in self.decoder_blocks:
            x = blk(x) # 消融实验
            # x = blk(x, scene_embed=scene_embed) # 原实验
        x = self.decoder_norm(x)

        # 预测器投影
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # 移除cls token

        return x

    def forward_loss(self, imgs, gradients, pred, mask):
        '''计算重建损失'''
        target = self.patchify(imgs) # 原始图像转换为目标patch序列
        if self.norm_pix_loss:
            # 像素归一化
            mean = target.mean(dim=-1, keepdim=True) 
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        
        if self.grad_weighted_loss:
            # 梯度加权：增强异常区域（梯度大的区域）的损失权重
            anomaly_map = imgs[:, 3:, :, :]
            anomaly_map = torch.clip(anomaly_map, min=0, max=1)
            # 设置异常图权重
            anomaly_weight = 0.3
            min_magnitude_anomaly = torch.ones((gradients.shape[0], 1, 1, 1), device=imgs.device) * 128
            anomaly_map *= torch.maximum(min_magnitude_anomaly, torch.amax(gradients, dim=(1, 2, 3), keepdim=True))
            gradients += anomaly_map * anomaly_weight
            
            grad_weights = F.max_pool2d(gradients, self.patch_size).mean(1)
            grad_weights = rearrange(grad_weights, 'b h w -> b (h w)')
            grad_weights = grad_weights / grad_weights.sum(dim=1, keepdims=True)
            loss = (loss * grad_weights).sum()
        else:
            # 仅计算被掩码补丁的损失（标准MAE损失）
            loss = (loss * mask).sum() / mask.sum()
            
        return loss

    def forward(self, imgs, targets, grad_mask=None, mask_ratio=0.75, scene_embed=None):
        '''模型整体前向传播'''
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, grad_mask, scene_embed=scene_embed)
        pred = self.forward_decoder(latent, ids_restore, scene_embed=scene_embed) # 解码器重建补丁
        loss = self.forward_loss(targets, grad_mask, pred, mask) # 计算损失
        
        if self.training:
            return loss, pred, mask
        else:
            # 测试时计算异常分数（原始与重建的差异）
            return loss, pred, mask, self.abnormal_score(targets, pred, mask, grad_mask)

    def abnormal_score(self, imgs, pred, mask, gradients):
        '''计算异常分数'''
        imgs = self.patchify(imgs)
        if self.use_only_masked_tokens_ab:
            mask = mask.bool()
            selected_pred = []
            selected_lbl = []
            for i in range(imgs.shape[0]):
                selected_pred.append(pred[i][mask[i]])
                selected_lbl.append(imgs[i][mask[i]])
            pred = torch.stack(selected_pred)
            imgs = torch.stack(selected_lbl)
        # 根据异常评分函数选择损失类型
        if self.abnormal_score_func == 'L1':
            # L1损失
            score = (imgs - pred).abs().mean((1, 2))
        else:  # 默认使用L2损失
            # L2损失（均方误差）
            score = ((imgs - pred) ** 2).mean((1, 2))
    
        return score

# 位置编码工具函数：生成2D正弦余弦位置编码
def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, cls_token=False):
    """
    生成2D正弦余弦位置嵌入
    grid_h: 高度方向的网格数
    grid_w: 宽度方向的网格数
    """
    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # 这里顺序是(w, h)
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_h.shape[0], grid_w.shape[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """从网格坐标生成2D正弦余弦位置嵌入"""
    assert embed_dim % 2 == 0
    
    # 使用半维度进行正弦余弦编码
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # 高度方向
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # 宽度方向
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """生成1D正弦余弦位置嵌入"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb