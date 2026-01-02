"""
Baseline Model - ResNet18 Pretrained

Modelo baseline simples usando ResNet18 com ImageNet weights.
"""

import logging
import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class BaselineModel(nn.Module):
    """
    ResNet18 Baseline para classificação binária PCam
    
    Arquitetura:
        Input (96x96x3)
        ↓
        ResNet18 Backbone (pretrained ImageNet)
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
        
        logger.info("Criando BaselineModel (ResNet18)")
        
        # ResNet18 backbone (sem classificador)
        self.backbone = timm.create_model(
            'resnet18',
            pretrained=model_config.get('pretrained', True),
            num_classes=0,  # Remove classificador
            global_pool='avg'  # Global average pooling
        )
        
        # ResNet18 features: 512 (não 2048 como ResNet50)
        backbone_features = 512
        hidden_dim = model_config.get('hidden_dim', 256)
        dropout = model_config.get('dropout', 0.5)
        
        # Classificador customizado
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 1)
        )
        
        logger.info(f"  Backbone: ResNet18 (pretrained={model_config.get('pretrained', True)})")
        logger.info(f"  Features: {backbone_features}")
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
        # Extrair features
        features = self.backbone(x)  # (B, 512)
        
        # Classificar
        logits = self.head(features)  # (B, 1)
        
        return logits
    
    def get_features(self, x):
        """
        Extrair features do backbone (útil para Grad-CAM)
        
        Args:
            x: Input tensor (B, 3, 96, 96)
            
        Returns:
            features: Feature maps (B, 512)
        """
        return self.backbone(x)
    
    def count_parameters(self):
        """Contar parâmetros treináveis"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


def create_baseline_model(config):
    """
    Factory function para criar modelo baseline
    
    Args:
        config: Dict com configuração
        
    Returns:
        model: BaselineModel instance
    """
    model = BaselineModel(config)
    
    # Log parâmetros
    params = model.count_parameters()
    logger.info(f"Modelo criado:")
    logger.info(f"  Total parameters: {params['total']:,}")
    logger.info(f"  Trainable parameters: {params['trainable']:,}")
    logger.info(f"  Frozen parameters: {params['frozen']:,}")
    
    return model


if __name__ == "__main__":
    # Teste do modelo
    import json
    
    print("🧪 Testando BaselineModel...")
    
    # Config dummy
    config = {
        'model': {
            'pretrained': True,
            'hidden_dim': 256,
            'dropout': 0.5
        }
    }
    
    # Criar modelo
    model = create_baseline_model(config)
    
    # Teste forward
    x = torch.randn(4, 3, 96, 96)
    logits = model(x)
    
    print(f"\n✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {logits.shape}")
    print(f"✅ Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Teste features
    features = model.get_features(x)
    print(f"✅ Features shape: {features.shape}")
    
    print("\n🎉 BaselineModel funciona!")
