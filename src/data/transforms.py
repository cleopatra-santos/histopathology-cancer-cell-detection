"""
Transformações e Augmentations para PCam
"""

import logging
import matplotlib.pyplot as plt
import cv2
import albumentations as A

# Setup logger
logger = logging.getLogger(__name__)


def get_train_transforms(config=None):
    """Transformações para TREINO (com augmentations)"""

    logger.debug("Criando train transforms")

    # Parâmetros default
    if config is None:
        aug_params = {
            'hflip': 0.5,
            'vflip': 0.5,
            'rotate': 45,
            'color_jitter': 0.2
        }
        logger.debug("Usando parâmetros default")
    else:
        aug_params = config.get('augmentation', {})
        logger.debug(f"Parâmetros: {aug_params}")

    transforms = A.Compose([
        A.HorizontalFlip(p=aug_params.get('hflip', 0.5)),
        A.VerticalFlip(p=aug_params.get('vflip', 0.5)),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.3, border_mode=cv2.BORDER_REFLECT),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=aug_params.get('color_jitter', 0.2)),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.1),
        A.ElasticTransform(alpha=1, sigma=50, p=0.1),
    ])

    logger.info("Train transforms criados")
    return transforms


def get_val_transforms(config=None):
    """Transformações para VALIDAÇÃO (sem augmentations)"""
    logger.info("Val transforms criados")
    return A.Compose([])


def get_tta_transforms():
    """Test-Time Augmentation"""
    tta = [
        A.Compose([]),
        A.Compose([A.HorizontalFlip(p=1.0)]),
        A.Compose([A.VerticalFlip(p=1.0)]),
        A.Compose([A.RandomRotate90(p=1.0)]),
    ]
    logger.info(f"TTA: {len(tta)} variações")
    return tta


def visualize_augmentations(image, num_samples=5):
    """Visualiza augmentations"""

    logger.info(f"Visualizando {num_samples} augmentations")

    transform = get_train_transforms()
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 3))

    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i in range(num_samples):
        augmented = transform(image=image)
        axes[i + 1].imshow(augmented['image'])
        axes[i + 1].set_title(f'Aug {i + 1}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    return fig