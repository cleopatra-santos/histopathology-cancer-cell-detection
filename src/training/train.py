"""
Script Principal de Treino

Treina modelo baseline ou híbrido no dataset PCam.

Usage:
    python scripts/train.py --config configs/baseline.json
    python scripts/train.py --config configs/hybrid.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Adicionar src ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import get_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.utils.logger import setup_logger


def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description='Treinar modelo PCam')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path para ficheiro de configuração JSON'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path para checkpoint para continuar treino'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu). Override config se fornecido'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Carregar configuração"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_model(config):
    """Criar modelo baseado na configuração"""
    architecture = config['model']['architecture']
    
    if architecture == 'resnet18':
        from src.models.baseline import create_baseline_model
        model = create_baseline_model(config)
    elif architecture == 'resnet18_cbam':
        from src.models.hybrid import create_hybrid_model
        model = create_hybrid_model(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def main():
    """Main training function"""
    # Parse args
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override device if provided
    if args.device:
        config['device'] = args.device
    
    # Setup logger
    exp_name = config.get('experiment_name', 'experiment')
    logger = setup_logger(exp_name, log_dir='logs')
    
    logger.info("="*70)
    logger.info(f"EXPERIMENTO: {exp_name}")
    logger.info("="*70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Description: {config.get('description', 'N/A')}")
    
    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed: {seed}")
    
    # Check CUDA
    device = config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA não disponível! Usando CPU")
        config['device'] = 'cpu'
        device = 'cpu'
    
    if device == 'cuda':
        logger.info(f"CUDA disponível: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Create transforms
    logger.info("\nCriando transformações...")
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    # Create dataloaders
    logger.info("\nCriando dataloaders...")
    train_loader, val_loader = get_dataloaders(
        config,
        train_transform,
        val_transform
    )
    
    # Create model
    logger.info("\nCriando modelo...")
    model = create_model(config)
    
    # Create trainer
    logger.info("\nCriando trainer...")
    from src.training.trainer import Trainer
    
    checkpoint_dir = config['checkpoints']['save_dir']
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=checkpoint_dir
    )
    
    # Resume if requested
    if args.resume:
        logger.info(f"\nCarregando checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    num_epochs = config['training']['epochs']
    logger.info(f"\nIniciando treino por {num_epochs} épocas...")
    
    try:
        trainer.train(num_epochs)
        logger.info("\n✅ Treino completo com sucesso!")
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Treino interrompido pelo utilizador")
        logger.info("Checkpoint salvo em: {checkpoint_dir}")
    except Exception as e:
        logger.error(f"\n❌ Erro durante treino: {e}")
        raise
    
    logger.info("\n" + "="*70)
    logger.info("FIM")
    logger.info("="*70)


if __name__ == "__main__":
    main()
