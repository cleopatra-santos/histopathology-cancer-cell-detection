"""
Hybrid Model - ResNet18 + CBAM

Modelo híbrido que combina CNN (ResNet18) com Attention (CBAM).
"""

import logging
import torch
import torch.nn as nn
import timm
from pathlib import Path
import sys

# Adicionar src ao path para importar CBAM
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.cbam import CBAM
except ImportError:
    # Se falhar, importar do mesmo diretório (para testes)
    from cbam import CBAM

logger = logging.getLogger(__name__)


class HybridModel(nn.Module):
    """
    ResNet18 + CBAM para classificação binária PCam
    
    Arquitetura:
        Input (96x96x3)
        ↓
        ResNet18 Layers 1-3 (pretrained)
        ↓
        ResNet18 Layer 4 + CBAM ← ATTENTION!
        ↓
        Global Average Pooling
        ↓
        FC: 512 → 256 → 1
        ↓
        Output (logit)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        model_config = config['model']
        
        logger.info("Criando HybridModel (ResNet18 + CBAM)")
        
        # Carregar ResNet18 pretrained
        backbone = timm.create_model(
            'resnet18',
            pretrained=model_config.get('pretrained', True),
            num_classes=0  # Sem classificador
        )
        
        # Extrair camadas da ResNet18
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels
        self.layer4 = backbone.layer4  # 512 channels
        
        # CBAM após layer4 (onde temos features ricas!)
        cbam_config = model_config.get('cbam', {})
        reduction_ratio = cbam_config.get('reduction_ratio', 16)
        kernel_size = cbam_config.get('kernel_size', 7)
        
        self.cbam = CBAM(
            in_channels=512,  # ResNet18 layer4 output
            reduction_ratio=reduction_ratio,
            kernel_size=kernel_size
        )
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classificador
        hidden_dim = model_config.get('hidden_dim', 256)
        dropout = model_config.get('dropout', 0.5)
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 1)
        )
        
        logger.info(f"  Backbone: ResNet18 (pretrained={model_config.get('pretrained', True)})")
        logger.info(f"  CBAM: reduction_ratio={reduction_ratio}, kernel_size={kernel_size}")
        logger.info(f"  Features: 512")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Dropout: {dropout}")
        
        # Inicializar head
        self._initialize_head()
    
    def _initialize_head(self):
        """Inicializar pesos do classificador"""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, 96, 96)
            
        Returns:
            logits: Output logits (B, 1)
        """
        # ResNet18 stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet18 blocks
        x = self.layer1(x)  # (B, 64, H/4, W/4)
        x = self.layer2(x)  # (B, 128, H/8, W/8)
        x = self.layer3(x)  # (B, 256, H/16, W/16)
        x = self.layer4(x)  # (B, 512, H/32, W/32) → (B, 512, 3, 3)
        
        # CBAM Attention! 🔥
        x = self.cbam(x)  # (B, 512, 3, 3) - COM ATENÇÃO!
        
        # Global pooling
        x = self.avgpool(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        
        # Classificar
        logits = self.head(x)  # (B, 1)
        
        return logits
    
    def get_features_and_attention(self, x):
        """
        Extrair features E mapas de atenção (para visualização)
        
        Args:
            x: Input tensor (B, 3, 96, 96)
            
        Returns:
            dict: {
                'features': (B, 512),
                'feature_maps': (B, 512, 3, 3),
                'channel_attention': (B, 512, 1, 1),
                'spatial_attention': (B, 1, 3, 3)
            }
        """
        # Forward até layer4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 512, 3, 3)
        
        # CBAM com attention maps
        attention_maps = self.cbam.get_attention_maps(x)
        
        # Pooling
        features = self.avgpool(attention_maps['output'])
        features = torch.flatten(features, 1)
        
        return {
            'features': features,
            'feature_maps': x,
            'channel_attention': attention_maps['channel_attention'],
            'spatial_attention': attention_maps['spatial_attention']
        }
    
    def count_parameters(self):
        """Contar parâmetros treináveis"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Contar CBAM separadamente
        cbam_params = sum(p.numel() for p in self.cbam.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'cbam': cbam_params
        }


def create_hybrid_model(config):
    """
    Factory function para criar modelo híbrido
    
    Args:
        config: Dict com configuração
        
    Returns:
        model: HybridModel instance
    """
    model = HybridModel(config)
    
    # Log parâmetros
    params = model.count_parameters()
    logger.info(f"Modelo criado:")
    logger.info(f"  Total parameters: {params['total']:,}")
    logger.info(f"  Trainable parameters: {params['trainable']:,}")
    logger.info(f"  CBAM parameters: {params['cbam']:,}")
    logger.info(f"  Frozen parameters: {params['frozen']:,}")
    
    return model


if __name__ == "__main__":
    # Teste do modelo
    import json
    
    print("🧪 Testando HybridModel...")
    
    # Config dummy
    config = {
        'model': {
            'pretrained': True,
            'hidden_dim': 256,
            'dropout': 0.5,
            'cbam': {
                'reduction_ratio': 16,
                'kernel_size': 7
            }
        }
    }
    
    # Criar modelo
    model = create_hybrid_model(config)
    
    # Teste forward
    x = torch.randn(4, 3, 96, 96)
    logits = model(x)
    
    print(f"\n✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {logits.shape}")
    print(f"✅ Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Teste features + attention
    result = model.get_features_and_attention(x)
    print(f"\n✅ Features shape: {result['features'].shape}")
    print(f"✅ Feature maps shape: {result['feature_maps'].shape}")
    print(f"✅ Channel attention shape: {result['channel_attention'].shape}")
    print(f"✅ Spatial attention shape: {result['spatial_attention'].shape}")
    
    print("\n🎉 HybridModel funciona!")
    print("\n💡 CBAM adiciona:")
    print("   • Channel Attention: O QUE é importante")
    print("   • Spatial Attention: ONDE é importante")
