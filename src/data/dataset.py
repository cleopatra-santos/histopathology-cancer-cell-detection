"""
PCam Dataset - PyTorch Dataset para PatchCamelyon
"""

import logging
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Setup logger para este módulo
logger = logging.getLogger(__name__)


class PCamDataset(Dataset):
    """
    Dataset para PatchCamelyon (PCam)

    Carrega imagens histológicas de 96x96 pixels e suas labels (tumor/normal).
    """

    def __init__(self, x_path, y_path, transform=None):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        logger.debug(f"Inicializando PCamDataset")
        logger.debug(f"  x_path: {self.x_path}")
        logger.debug(f"  y_path: {self.y_path}")

        # Verificar se ficheiros existem
        if not self.x_path.exists():
            logger.error(f"Ficheiro não encontrado: {self.x_path}")
            raise FileNotFoundError(f"Ficheiro não encontrado: {self.x_path}")
        if not self.y_path.exists():
            logger.error(f"Ficheiro não encontrado: {self.y_path}")
            raise FileNotFoundError(f"Ficheiro não encontrado: {self.y_path}")

        logger.debug("Ficheiros encontrados")

        # Obter tamanho do dataset
        with h5py.File(self.x_path, 'r') as f:
            self.length = f['x'].shape[0]

        logger.info(f"Dataset criado: {self.length:,} imagens")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Carregar imagem e label
        with h5py.File(self.x_path, 'r') as fx:
            image = fx['x'][idx]

        with h5py.File(self.y_path, 'r') as fy:
            label = fy['y'][idx, 0, 0, 0]

        # Aplicar transformações
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Converter para tensor e normalizar
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


class PCamDatasetCached(Dataset):
    """
    Versão com cache em memória (para datasets pequenos)
    """

    def __init__(self, x_path, y_path, transform=None):
        self.transform = transform

        logger.info("Carregando dataset na memória...")

        with h5py.File(x_path, 'r') as fx:
            self.images = fx['x'][:]
            logger.debug(f"Imagens carregadas: {self.images.shape}")

        with h5py.File(y_path, 'r') as fy:
            self.labels = fy['y'][:, 0, 0, 0]
            logger.debug(f"Labels carregadas: {self.labels.shape}")

        logger.info(f"Dataset em cache: {len(self.images):,} imagens")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


def get_dataloaders(config, train_transform=None, val_transform=None):
    """Cria DataLoaders para treino e validação"""
    logger.info("Criando datasets...")

    try:
        train_dataset = PCamDataset(
            config['data']['train_x'],
            config['data']['train_y'],
            transform=train_transform
        )

        val_dataset = PCamDataset(
            config['data']['valid_x'],
            config['data']['valid_y'],
            transform=val_transform
        )

        logger.info("Criando dataloaders...")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 4),
            pin_memory=True,
            drop_last=False
        )

        logger.info(f"Train loader: {len(train_loader)} batches")
        logger.info(f"Val loader: {len(val_loader)} batches")

        return train_loader, val_loader

    except Exception as e:
        logger.error(f"Erro ao criar dataloaders: {e}")
        raise