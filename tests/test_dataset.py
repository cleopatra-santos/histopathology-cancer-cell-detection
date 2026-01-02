"""
Testes para dataset.py

"""

import sys
import logging
from pathlib import Path

import pytest
import torch

from src.data.dataset import (
    PCamDataset,
    PCamDatasetCached,
    get_dataloaders
)
from src.data.transforms import (
    get_train_transforms,
    get_val_transforms
)

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

TRAIN_X = DATA_ROOT / "camelyonpatch_level_2_split_train_x.h5"
TRAIN_Y = DATA_ROOT / "camelyonpatch_level_2_split_train_y.h5"
VALID_X = DATA_ROOT / "camelyonpatch_level_2_split_valid_x.h5"
VALID_Y = DATA_ROOT / "camelyonpatch_level_2_split_valid_y.h5"


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def check_files_exist():
    """Verifica se os ficheiros do dataset existem"""
    logger.info("Verificando ficheiros do dataset")

    files = [TRAIN_X, TRAIN_Y, VALID_X, VALID_Y]
    missing = [f for f in files if not f.exists()]

    if missing:
        for f in missing:
            logger.error(f"Ficheiro não encontrado: {f}")
        pytest.fail("Dataset não encontrado em data/raw/")

    logger.info("Todos os ficheiros encontrados")
    return True


@pytest.fixture
def train_dataset(check_files_exist):
    return PCamDataset(str(TRAIN_X), str(TRAIN_Y))


@pytest.fixture
def val_dataset(check_files_exist):
    return PCamDataset(str(VALID_X), str(VALID_Y))


@pytest.fixture
def train_transform():
    return get_train_transforms()


@pytest.fixture
def val_transform():
    return get_val_transforms()


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def test_pcam_dataset_basic(train_dataset):
    """Teste básico do PCamDataset"""
    logger.info("Teste básico do dataset")

    assert len(train_dataset) > 0

    image, label = train_dataset[0]

    assert image.shape == (3, 96, 96)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    assert image.min() >= 0.0
    assert image.max() <= 1.0

    assert label.item() in [0, 1]


def test_pcam_dataset_multiple_samples(train_dataset):
    """Teste com múltiplos índices"""
    logger.info("Teste múltiplas amostras")

    indices = [0, 100, 1000, 10000]

    for idx in indices:
        image, label = train_dataset[idx]
        assert image.shape == (3, 96, 96)
        assert label.item() in [0, 1]


def test_pcam_dataset_with_transforms(train_transform):
    """Teste augmentations"""
    logger.info("Teste dataset com transforms")

    dataset = PCamDataset(
        str(TRAIN_X),
        str(TRAIN_Y),
        transform=train_transform
    )

    image1, _ = dataset[0]
    image2, _ = dataset[0]

    assert not torch.allclose(image1, image2)


def test_pcam_dataset_cached(check_files_exist):
    """Teste PCamDatasetCached"""
    logger.info("Teste dataset cached")

    dataset = PCamDatasetCached(str(VALID_X), str(VALID_Y))

    assert len(dataset) > 0

    image, label = dataset[0]
    assert image.shape == (3, 96, 96)
    assert label.item() in [0, 1]


def test_dataloader_batches(val_dataset):
    """Teste DataLoader"""
    logger.info("Teste DataLoader")

    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    images, labels = next(iter(loader))

    assert images.shape == (32, 3, 96, 96)
    assert labels.shape == (32,)


def test_get_dataloaders_function(train_transform, val_transform):
    """Teste helper get_dataloaders"""
    logger.info("Teste get_dataloaders")

    config = {
        "data": {
            "train_x": str(TRAIN_X),
            "train_y": str(TRAIN_Y),
            "valid_x": str(VALID_X),
            "valid_y": str(VALID_Y),
            "batch_size": 32,
            "num_workers": 0
        }
    }

    train_loader, val_loader = get_dataloaders(
        config,
        train_transform,
        val_transform
    )

    assert train_loader is not None
    assert val_loader is not None

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert len(train_batch) == 2
    assert len(val_batch) == 2
