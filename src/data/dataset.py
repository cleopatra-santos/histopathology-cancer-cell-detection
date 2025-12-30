#!/usr/bin/env python3
"""
Download PatchCamelyon (PCam) Dataset
Versão SIMPLIFICADA e CORRIGIDA

Usage:
    python scripts/download_pcam.py [--mode minimal|full]
"""

import argparse
import urllib.request
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np


class DownloadProgressBar(tqdm):
    """Progress bar para downloads"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class PCamDownloader:
    """
    Downloader para o dataset PatchCamelyon
    """

    def __init__(self):
        # Path absoluto - funciona de qualquer lugar
        project_root = Path(__file__).parent.parent  # scripts/ -> raiz/
        self.data_dir = project_root / "data" / "raw"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = "https://zenodo.org/records/2546921/files"

        self.files = {
            'train_x': 'camelyonpatch_level_2_split_train_x.h5.gz',
            'train_y': 'camelyonpatch_level_2_split_train_y.h5.gz',
            'valid_x': 'camelyonpatch_level_2_split_valid_x.h5.gz',
            'valid_y': 'camelyonpatch_level_2_split_valid_y.h5.gz',
            'test_x': 'camelyonpatch_level_2_split_test_x.h5.gz',
            'test_y': 'camelyonpatch_level_2_split_test_y.h5.gz',
        }

        # Tamanhos aproximados (GB)
        self.sizes = {
            'train_x': 6.8,
            'train_y': 0.05,
            'valid_x': 1.2,
            'valid_y': 0.01,
            'test_x': 1.2,
            'test_y': 0.01,
        }

    def download_file(self, key):
        """Download single file"""
        filename = self.files[key]
        url = f"{self.base_url}/{filename}"
        output_path = self.data_dir / filename

        if output_path.exists():
            print(f"✅ {filename} já existe, pulando...")
            return True

        size_gb = self.sizes.get(key, 0)
        print(f"\n📥 Baixando {key}...")
        print(f"   URL: {url}")
        print(f"   Tamanho: ~{size_gb} GB")

        try:
            with DownloadProgressBar(
                    unit='B',
                    unit_scale=True,
                    miniters=1,
                    desc=f"  {key}"
            ) as pbar:
                urllib.request.urlretrieve(
                    url,
                    output_path,
                    reporthook=pbar.update_to
                )
            print(f"✅ Download completo: {filename}")
            return True

        except Exception as e:
            print(f"❌ Erro ao baixar {filename}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False

    def extract_file(self, key):
        """Extract .gz file"""
        gz_filename = self.files[key]
        gz_path = self.data_dir / gz_filename
        h5_path = gz_path.with_suffix('')

        if h5_path.exists():
            print(f"✅ {h5_path.name} já existe")
            return True

        if not gz_path.exists():
            print(f"❌ {gz_filename} não encontrado!")
            return False

        print(f"\n📦 Extraindo {gz_filename}...")

        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(h5_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            print(f"✅ Extraído: {h5_path.name}")

            # Opcional: remover .gz para economizar espaço
            response = input(f"   Remover {gz_filename}? (s/N): ")
            if response.lower() == 's':
                gz_path.unlink()
                print(f"   🗑️  Removido {gz_filename}")

            return True

        except Exception as e:
            print(f"❌ Erro ao extrair {gz_filename}: {e}")
            if h5_path.exists():
                h5_path.unlink()
            return False

    def verify_dataset(self):
        """Verify downloaded dataset"""
        print("\n🔍 Verificando dataset...")
        print("=" * 70)

        required = ['train_x', 'train_y', 'valid_x', 'valid_y']

        all_ok = True
        for key in required:
            h5_file = self.data_dir / f"camelyonpatch_level_2_split_{key}.h5"

            if h5_file.exists():
                size_mb = h5_file.stat().st_size / (1024 ** 2)
                print(f"✅ {h5_file.name:45s} ({size_mb:,.0f} MB)")
            else:
                print(f"❌ {h5_file.name:45s} - NÃO ENCONTRADO")
                all_ok = False

        return all_ok

    def show_dataset_info(self):
        """Show dataset information"""
        try:
            print("\n📊 INFORMAÇÕES DO DATASET")
            print("=" * 70)

            train_x = self.data_dir / "camelyonpatch_level_2_split_train_x.h5"
            train_y = self.data_dir / "camelyonpatch_level_2_split_train_y.h5"

            if not (train_x.exists() and train_y.exists()):
                print("⚠️  Arquivos não encontrados para análise")
                return

            with h5py.File(train_x, 'r') as fx:
                x_shape = fx['x'].shape
                print(f"Training Images:")
                print(f"  Shape: {x_shape}")
                print(f"  Count: {x_shape[0]:,}")
                print(f"  Size: {x_shape[1]}×{x_shape[2]}×{x_shape[3]}")

            with h5py.File(train_y, 'r') as fy:
                y_data = fy['y'][:]
                positive = np.sum(y_data)
                negative = len(y_data) - positive

                print(f"\nTraining Labels:")
                print(f"  Total: {len(y_data):,}")
                print(f"  Tumor: {positive:,} ({positive / len(y_data) * 100:.1f}%)")
                print(f"  Normal: {negative:,} ({negative / len(y_data) * 100:.1f}%)")

            print("\n✅ Dataset pronto para uso!")

        except ImportError:
            print("\n💡 Instale h5py para ver informações:")
            print("   poetry add h5py")
        except Exception as e:
            print(f"\n⚠️  Erro ao analisar dataset: {e}")

    def download_minimal(self):
        """Download minimal (train + valid only)"""
        print("🚀 DOWNLOAD MÍNIMO")
        print("=" * 70)
        print("Baixando: train_x, train_y, valid_x, valid_y")
        print("Total estimado: ~8 GB")
        print("Tempo estimado: 20-40 min (depende da conexão)")
        print("=" * 70)

        files_to_download = ['train_x', 'train_y', 'valid_x', 'valid_y']

        for key in files_to_download:
            if not self.download_file(key):
                print(f"\n❌ Falha no download de {key}")
                return False

            if not self.extract_file(key):
                print(f"\n❌ Falha na extração de {key}")
                return False

        return True

    def download_full(self):
        """Download full dataset (train + valid + test)"""
        print("🚀 DOWNLOAD COMPLETO")
        print("=" * 70)
        print("Baixando: train, valid, test")
        print("Total estimado: ~10 GB")
        print("Tempo estimado: 30-50 min (depende da conexão)")
        print("=" * 70)

        for key in self.files.keys():
            if not self.download_file(key):
                print(f"\n❌ Falha no download de {key}")
                return False

            if not self.extract_file(key):
                print(f"\n❌ Falha na extração de {key}")
                return False

        return True

    def run(self, mode='minimal'):
        """Run download pipeline"""
        print("\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 15 + "🔬 PCam Dataset Downloader" + " " * 26 + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        print(f"📁 Diretório de destino: {self.data_dir.absolute()}")
        print()

        # Download
        if mode == 'minimal':
            success = self.download_minimal()  # ✅ Corrigido
        else:
            success = self.download_full()

        if not success:
            print("\n❌ Download incompleto!")
            return False

        # Verify
        if not self.verify_dataset():
            print("\n⚠️  Verificação falhou!")
            return False

        # Show info
        self.show_dataset_info()

        print("\n" + "=" * 70)
        print("✅ DOWNLOAD COMPLETO!")
        print("=" * 70)
        print("\n📝 Próximos passos:")
        print("1. Explorar dados: jupyter notebook notebooks/01_exploration.ipynb")
        print("2. Treinar modelo: python scripts/train.py --config configs/baseline.json")
        print()

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Download PCam dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download mínimo (recomendado para 1 semana)
  python scripts/download_pcam.py --mode minimal

  # Download completo (inclui test set)
  python scripts/download_pcam.py --mode full
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['minimal', 'full'],
        default='minimal',
        help='Modo de download (default: minimal)'
    )

    args = parser.parse_args()

    # Create downloader (sem argumentos!)
    downloader = PCamDownloader()  # ✅ Corrigido

    # Run
    success = downloader.run(mode=args.mode)

    if success:
        print("🎉 Sucesso!")
        exit(0)
    else:
        print("💥 Falha!")
        exit(1)


if __name__ == "__main__":
    main()