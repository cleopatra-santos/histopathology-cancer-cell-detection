"""
Sistema de Logging para o Projeto
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_dir='logs', level=logging.INFO):
    """
    Configura logger para o projeto

    Args:
        name: Nome do logger (ex: 'train', 'eval')
        log_dir: Diretório para salvar logs
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Logger configurado
    """
    # Criar diretório de logs
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Nome do ficheiro com timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'{name}_{timestamp}.log'

    # Criar logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remover handlers existentes (evitar duplicados)
    logger.handlers.clear()

    # Formato
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para ficheiro
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger '{name}' configurado")
    logger.info(f"Log file: {log_file}")

    return logger


def get_logger(name):
    """
    Obtém logger existente ou cria novo
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        return setup_logger(name)

    return logger


class TrainingLogger:
    """
    Logger especializado para treino

    Mantém histórico de métricas e facilita logging durante treino.
    """

    def __init__(self, name, log_dir='logs'):
        self.logger = setup_logger(name, log_dir)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
        }

    def log_epoch(self, epoch, **metrics):
        """
        Log métricas de uma época

        Args:
            epoch: Número da época
            **metrics: train_loss, val_loss, val_acc, etc.
        """
        msg = f"Epoch {epoch:03d}"

        for key, value in metrics.items():
            msg += f" | {key}: {value:.4f}"

            # Adicionar ao histórico
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        self.logger.info(msg)

    def log_metrics(self, metrics, prefix=''):
        """
        Log dicionário de métricas
        """
        msg = "Metrics"
        if prefix:
            msg = f"{prefix.capitalize()} metrics"

        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.4f}"
            else:
                msg += f" | {key}: {value}"

        self.logger.info(msg)

    def log_model_info(self, model, device):
        """Log informações do modelo"""
        import torch

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info("=" * 70)
        self.logger.info("MODEL INFO")
        self.logger.info(f"Architecture: {model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Device: {device}")
        self.logger.info("=" * 70)

    def log_config(self, config):
        """Log configuração do experimento"""
        import json

        self.logger.info("=" * 70)
        self.logger.info("CONFIGURATION")
        self.logger.info(json.dumps(config, indent=2))
        self.logger.info("=" * 70)

    def save_history(self, path):
        """Salvar histórico de métricas"""
        import json

        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

        self.logger.info(f"History saved to {path}")

    def info(self, msg):
        """Atalho para self.logger.info"""
        self.logger.info(msg)

    def warning(self, msg):
        """Atalho para self.logger.warning"""
        self.logger.warning(msg)

    def error(self, msg):
        """Atalho para self.logger.error"""
        self.logger.error(msg)


# Logger global para uso rápido
_default_logger = None


def log(msg, level='info'):
    """
    Função de logging rápida

    Args:
        msg: Mensagem
        level: 'info', 'warning', 'error'
    """
    global _default_logger

    if _default_logger is None:
        _default_logger = setup_logger('main')

    if level == 'info':
        _default_logger.info(msg)
    elif level == 'warning':
        _default_logger.warning(msg)
    elif level == 'error':
        _default_logger.error(msg)


if __name__ == "__main__":
    # Teste do sistema de logging
    print("\n Testando sistema de logging...\n")

    # Teste 1: Logger básico
    print("=" * 70)
    print("TEST 1: Logger Básico")
    print("=" * 70)
    logger = setup_logger('test')
    logger.info("Mensagem de info")
    logger.warning("Mensagem de warning")
    logger.error("Mensagem de erro")

    # Teste 2: TrainingLogger
    print("\n" + "=" * 70)
    print("TEST 2: Training Logger")
    print("=" * 70)
    train_logger = TrainingLogger('train_test')
    train_logger.log_epoch(1, train_loss=0.5, val_loss=0.4, val_acc=0.85)
    train_logger.log_epoch(2, train_loss=0.3, val_loss=0.35, val_acc=0.88)
    train_logger.log_metrics({'precision': 0.9, 'recall': 0.85, 'f1': 0.87})

    # Teste 3: Função rápida
    print("\n" + "=" * 70)
    print("TEST 3: Quick Log Function")
    print("=" * 70)
    log("Teste de logging rápido")
    log("Aviso de teste", level='warning')
    log("Erro de teste", level='error')

    print("\n" + "=" * 70)
    print(" TODOS OS TESTES PASSARAM!")
    print(f" Logs salvos em: {Path('logs').absolute()}")
    print("=" * 70)