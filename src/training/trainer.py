"""
Trainer - Loop de Treino e Validação

Classe completa para treinar modelos de classificação binária.
"""

import logging
import time
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer para modelos de classificação binária
    
    Features:
        • Mixed precision training
        • Early stopping
        • Learning rate scheduling
        • Checkpointing
        • Logging completo
    """
    
    def __init__(self, model, train_loader, val_loader, config, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device(config.get('device', 'cuda'))
        self.model.to(self.device)
        
        logger.info(f"Trainer inicializado")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Save dir: {self.save_dir}")
        
        # Setup training components
        self._setup_loss()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_mixed_precision()
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc_roc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
    def _setup_loss(self):
        """Setup loss function"""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'bce_with_logits')
        
        if loss_type == 'bce_with_logits':
            pos_weight = loss_config.get('pos_weight', 1.0)
            if pos_weight != 1.0:
                pos_weight = torch.tensor([pos_weight]).to(self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        logger.info(f"  Loss: {loss_type}")
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        train_config = self.config['training']
        opt_type = train_config.get('optimizer', 'adam').lower()
        lr = train_config.get('learning_rate', 1e-4)
        weight_decay = train_config.get('weight_decay', 1e-4)
        
        if opt_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        logger.info(f"  Optimizer: {opt_type}, lr={lr}, wd={weight_decay}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        train_config = self.config['training']
        sched_type = train_config.get('scheduler', 'reduce_on_plateau')
        
        if sched_type == 'reduce_on_plateau':
            sched_params = train_config.get('scheduler_params', {})
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_params.get('mode', 'max'),
                factor=sched_params.get('factor', 0.5),
                patience=sched_params.get('patience', 3),
                min_lr=sched_params.get('min_lr', 1e-7)
            )
        elif sched_type == 'cosine':
            epochs = train_config.get('epochs', 30)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs
            )
        elif sched_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        logger.info(f"  Scheduler: {sched_type}")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        use_amp = self.config['training'].get('mixed_precision', False)
        
        if use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler()
            self.use_amp = True
            logger.info("  Mixed precision: ENABLED")
        else:
            self.scaler = None
            self.use_amp = False
            logger.info("  Mixed precision: DISABLED")
    
    def train_epoch(self, epoch):
        """
        Treinar uma época
        
        Args:
            epoch: Número da época
            
        Returns:
            float: Loss média da época
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass com mixed precision
            if self.use_amp:
                with autocast():
                    logits = self.model(images).squeeze()
                    loss = self.criterion(logits, labels)
                
                # Backward com scaler
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images).squeeze()
                loss = self.criterion(logits, labels)
                
                # Backward normal
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Média
        avg_loss = epoch_loss / num_batches
        
        return avg_loss
    
    def validate(self):
        """
        Validar modelo
        
        Returns:
            dict: Métricas de validação
        """
        from metrics import MetricsCalculator
        
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        metrics_calc = MetricsCalculator(threshold=0.5)
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                if self.use_amp:
                    with autocast():
                        logits = self.model(images).squeeze()
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images).squeeze()
                    loss = self.criterion(logits, labels)
                
                val_loss += loss.item()
                
                # Update metrics
                metrics_calc.update(logits.unsqueeze(1), labels)
        
        # Calcular métricas
        metrics = metrics_calc.compute()
        metrics['loss'] = val_loss / num_batches
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        Salvar checkpoint
        
        Args:
            epoch: Número da época
            metrics: Dict com métricas
            is_best: Se é o melhor modelo até agora
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Salvar último
        last_path = self.save_dir / 'last.pth'
        torch.save(checkpoint, last_path)
        
        # Salvar melhor
        if is_best:
            best_path = self.save_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"  💾 Best model saved: {best_path}")
    
    def load_checkpoint(self, path):
        """
        Carregar checkpoint
        
        Args:
            path: Path para o checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Checkpoint loaded from: {path}")
        logger.info(f"  Epoch: {checkpoint['epoch']}")
        logger.info(f"  Metrics: {checkpoint['metrics']}")
    
    def train(self, num_epochs):
        """
        Loop de treino completo
        
        Args:
            num_epochs: Número de épocas
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"INICIANDO TREINO")
        logger.info(f"{'='*70}")
        logger.info(f"Épocas: {num_epochs}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        
        # Early stopping config
        early_stop_config = self.config['training'].get('early_stopping', {})
        use_early_stop = early_stop_config.get('enabled', True)
        patience = early_stop_config.get('patience', 7)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc_roc'].append(val_metrics['auc_roc'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Log
            epoch_time = time.time() - epoch_start
            logger.info(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.1f}s")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss:   {val_metrics['loss']:.4f}")
            logger.info(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val Prec:   {val_metrics['precision']:.4f}")
            logger.info(f"  Val Rec:    {val_metrics['recall']:.4f}")
            logger.info(f"  Val F1:     {val_metrics['f1']:.4f}")
            logger.info(f"  Val AUC:    {val_metrics['auc_roc']:.4f}")
            logger.info(f"  LR:         {current_lr:.2e}")
            
            # Check if best
            current_metric = val_metrics['accuracy']
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                logger.info(f"  🌟 New best accuracy: {self.best_metric:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if use_early_stop and self.epochs_without_improvement >= patience:
                logger.info(f"\n⚠️  Early stopping triggered!")
                logger.info(f"  No improvement for {patience} epochs")
                logger.info(f"  Best epoch: {self.best_epoch}")
                logger.info(f"  Best accuracy: {self.best_metric:.4f}")
                break
        
        # Treino completo
        total_time = time.time() - start_time
        logger.info(f"\n{'='*70}")
        logger.info(f"TREINO COMPLETO")
        logger.info(f"{'='*70}")
        logger.info(f"Tempo total: {total_time/60:.1f} min")
        logger.info(f"Melhor época: {self.best_epoch}")
        logger.info(f"Melhor accuracy: {self.best_metric:.4f}")
        
        # Salvar history
        history_path = self.save_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"History saved: {history_path}")
