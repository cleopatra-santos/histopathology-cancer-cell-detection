"""
CBAM - Convolutional Block Attention Module

Implementação do CBAM para atenção em CNNs.
Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    """
    Canal Attention Module
    
    Foca em "O QUE é importante?" - Quais filtros/canais são mais relevantes.
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            attention: (B, C, 1, 1)
        """
        # Average pooling path
        avg_out = self.mlp(self.avg_pool(x))
        
        # Max pooling path
        max_out = self.mlp(self.max_pool(x))
        
        # Combine
        attention = self.sigmoid(avg_out + max_out)
        
        return attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    
    Foca em "ONDE é importante?" - Quais localizações espaciais são relevantes.
    """
    
    def __init__(self, kernel_size=7):
        super().__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            attention: (B, 1, H, W)
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # Convolution
        attention = self.sigmoid(self.conv(combined))  # (B, 1, H, W)
        
        return attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    
    Combina Channel Attention e Spatial Attention sequencialmente.
    
    Args:
        in_channels: Número de canais de entrada
        reduction_ratio: Ratio de redução para channel attention
        kernel_size: Tamanho do kernel para spatial attention
    
    Example:
        >>> cbam = CBAM(512, reduction_ratio=16, kernel_size=7)
        >>> x = torch.randn(4, 512, 3, 3)
        >>> out = cbam(x)
        >>> print(out.shape)  # (4, 512, 3, 3)
    """
    
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        logger.debug(f"CBAM criado: channels={in_channels}, "
                    f"reduction={reduction_ratio}, kernel={kernel_size}")
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W) - Features com atenção aplicada
        """
        # Channel attention
        channel_att = self.channel_attention(x)  # (B, C, 1, 1)
        x = x * channel_att  # Broadcasting: (B, C, H, W) * (B, C, 1, 1)
        
        # Spatial attention
        spatial_att = self.spatial_attention(x)  # (B, 1, H, W)
        x = x * spatial_att  # Broadcasting: (B, C, H, W) * (B, 1, H, W)
        
        return x
    
    def get_attention_maps(self, x):
        """
        Extrair mapas de atenção para visualização
        
        Args:
            x: (B, C, H, W)
            
        Returns:
            dict: {
                'channel_attention': (B, C, 1, 1),
                'spatial_attention': (B, 1, H, W),
                'output': (B, C, H, W)
            }
        """
        # Channel attention
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        # Spatial attention
        spatial_att = self.spatial_attention(x_channel)
        x_out = x_channel * spatial_att
        
        return {
            'channel_attention': channel_att,
            'spatial_attention': spatial_att,
            'output': x_out
        }


if __name__ == "__main__":
    # Teste do CBAM
    print("🧪 Testando CBAM...")
    
    # Criar módulo
    cbam = CBAM(in_channels=512, reduction_ratio=16, kernel_size=7)
    
    # Input dummy (batch=4, channels=512, spatial=3x3)
    x = torch.randn(4, 512, 3, 3)
    
    print(f"\n✅ Input shape: {x.shape}")
    
    # Forward pass
    out = cbam(x)
    print(f"✅ Output shape: {out.shape}")
    
    # Attention maps
    attention = cbam.get_attention_maps(x)
    print(f"✅ Channel attention shape: {attention['channel_attention'].shape}")
    print(f"✅ Spatial attention shape: {attention['spatial_attention'].shape}")
    
    # Verificar que output tem mesma shape
    assert out.shape == x.shape, "Output shape deve ser igual ao input!"
    
    print("\n🎉 CBAM funciona!")
    print("\nCanal Attention responde: 'O QUE é importante?'")
    print("Spatial Attention responde: 'ONDE é importante?'")
