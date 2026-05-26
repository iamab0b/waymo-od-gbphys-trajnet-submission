"""
Full training script for GB-Phys TrajNet.

HPC-compatible (SLURM); no GUI, all paths via argparse.
Uses TensorFlow tf.data for data loading and PyTorch for model / optimisation.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SLURM

import argparse
import os
import sys
import time
import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from src.data.loader import (
    create_gcs_dataset,
    create_local_dataset,
    tf_to_torch_batch,
    GCS_TRAIN,
    GCS_VAL,
    GCS_TEST,
    GCS_VAL_INTERACTIVE,
    GCS_TEST_INTERACTIVE,
    GCS_SPLIT_PATTERNS,
)
from src.data.preprocessing import (
    random_rotation_augmentation,
    random_translation_augmentation,
)
from src.models.gbphys_trajnet import GBPhysTrajNet
from src.losses import CombinedLoss
from src.training.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    save_best_checkpoint,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='GB-Phys TrajNet trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    data = parser.add_argument_group('Data')
    data.add_argument('--data_dir', type=str, default=None,
                      help='Local directory with training TFRecord shards')
    data.add_argument('--gcs_train', type=str, default=GCS_TRAIN,
                      help='GCS glob pattern for training data')
    data.add_argument('--gcs_val', type=str, default=GCS_VAL,
                      help='GCS glob pattern for validation data')
    data.add_argument('--gcs_test', type=str, default=GCS_TEST,
                      help='GCS glob pattern for test data (for final eval)')
    data.add_argument('--train_split', type=str, default='train',
                      choices=['train', 'train_20s'],
                      help='Training split to use (train=9.1s, train_20s=20s scenarios)')
    data.add_argument('--val_split', type=str, default='val',
                      choices=['val', 'val_interactive'],
                      help='Validation split to use')
    data.add_argument('--val_data_dir', type=str, default=None,
                      help='Local directory for validation data')
    data.add_argument('--subset_shards', type=int, default=None,
                      help='Limit training to first N shards (debug)')
    data.add_argument('--shuffle_buffer', type=int, default=1000,
                      help='tf.data shuffle buffer size')

    # Training
    train = parser.add_argument_group('Training')
    train.add_argument('--num_epochs', type=int, default=30)
    train.add_argument('--batch_size', type=int, default=32)
    train.add_argument('--num_workers', type=int, default=4,
                       help='Number of CPU workers for data loading')
    train.add_argument('--lr', type=float, default=5e-4)
    train.add_argument('--weight_decay', type=float, default=1e-4)
    train.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                       help='Training phase: 1=Stage1, 2=Stage2, 3=end-to-end')
    train.add_argument('--accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps (effective bs = bs * accum)')
    train.add_argument('--use_mixed_precision', action='store_true',
                       help='Use FP16 mixed precision training')
    train.add_argument('--clip_grad_norm', type=float, default=5.0,
                       help='Gradient clipping max norm (0 = disabled)')
    train.add_argument('--augment', action='store_true',
                       help='Enable random rotation/translation augmentation')

    # Model
    model = parser.add_argument_group('Model')
    model.add_argument('--use_transformer_encoder', action='store_true',
                       help='Use Transformer instead of CNN for agent encoder')
    model.add_argument('--agent_embedding_dim', type=int, default=128)
    model.add_argument('--road_embedding_dim',  type=int, default=128)
    model.add_argument('--gru_hidden_dim',       type=int, default=256)
    model.add_argument('--gru_layers',           type=int, default=2)
    model.add_argument('--K',                    type=int, default=6)
    model.add_argument('--N_candidates',         type=int, default=64)

    # Losses
    loss_grp = parser.add_argument_group('Losses')
    loss_grp.add_argument('--lambda_goal',     type=float, default=1.0)
    loss_grp.add_argument('--lambda_wta',      type=float, default=1.0)
    loss_grp.add_argument('--lambda_physics',  type=float, default=0.1)
    loss_grp.add_argument('--lambda_boundary', type=float, default=0.5)

    # Checkpointing / logging
    io_grp = parser.add_argument_group('IO')
    io_grp.add_argument('--checkpoint_dir', type=str, default='results/checkpoints',
                        help='Directory to save checkpoints')
    io_grp.add_argument('--log_dir', type=str, default='results/logs/tensorboard',
                        help='TensorBoard log directory')
    io_grp.add_argument('--resume_from', type=str, default=None,
                        help='Path to a checkpoint to resume from')
    io_grp.add_argument('--log_every',  type=int, default=50,
                        help='Log training metrics every N steps')
    io_grp.add_argument('--save_every', type=int, default=500,
                        help='Save checkpoint every N steps')
    io_grp.add_argument('--eval_every', type=int, default=1000,
                        help='Run validation every N steps')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42)

    return parser


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training loop for GB-Phys TrajNet.

    Supports:
      - Three-phase training with parameter freezing
      - Mixed-precision (torch.cuda.amp)
      - Gradient accumulation
      - Per-step and per-epoch checkpointing
      - TensorBoard logging
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device
                                   if torch.cuda.is_available() or args.device == 'cpu'
                                   else 'cpu')
        logger.info(f'Using device: {self.device}')

        torch.manual_seed(args.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)

        # Model
        self.model = self._build_model()
        self.model.to(self.device)
        logger.info(f'Model parameters: {self.model.count_parameters():,}')

        # Loss
        self.criterion = CombinedLoss(
            lambda_goal=args.lambda_goal,
            lambda_wta=args.lambda_wta,
            lambda_physics=args.lambda_physics,
            lambda_boundary=args.lambda_boundary,
        )

        # Optimizer (built after phase-based freezing)
        self._apply_phase_freezing()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Scaler for mixed precision
        self.scaler = GradScaler(enabled=args.use_mixed_precision and
                                         self.device.type == 'cuda')

        # State
        self.epoch        = 0
        self.global_step  = 0
        self.best_metric  = float('inf')

        # TensorBoard
        self.writer = None
        if TB_AVAILABLE:
            os.makedirs(args.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=args.log_dir)

        # Resume
        if args.resume_from:
            self._resume(args.resume_from)
        else:
            latest = find_latest_checkpoint(args.checkpoint_dir)
            if latest:
                logger.info(f'Found existing checkpoint: {latest}')
                self._resume(latest)

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> GBPhysTrajNet:
        cfg = {
            'agent_embedding_dim':     self.args.agent_embedding_dim,
            'road_embedding_dim':      self.args.road_embedding_dim,
            'gru_hidden_dim':          self.args.gru_hidden_dim,
            'gru_layers':              self.args.gru_layers,
            'K':                       self.args.K,
            'N_candidates':            self.args.N_candidates,
            'use_transformer_encoder': self.args.use_transformer_encoder,
        }
        return GBPhysTrajNet(config=cfg)

    def _apply_phase_freezing(self):
        """Freeze parameters appropriate for the current training phase."""
        if self.args.phase == 2:
            self.model.freeze_stage1()
            logger.info('Phase 2: Stage-1 parameters frozen.')
        elif self.args.phase == 3:
            self.model.unfreeze_all()
            logger.info('Phase 3: All parameters trainable.')
        else:
            logger.info('Phase 1: Stage-1 only training.')

    def _build_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return optim.AdamW(trainable, lr=self.args.lr,
                           weight_decay=self.args.weight_decay)

    def _build_scheduler(self):
        # Cosine annealing with warm restarts
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.num_epochs,
            eta_min=self.args.lr * 0.01,
        )

    def _resume(self, path: str):
        self.epoch, self.global_step, _ = load_checkpoint(
            self.model, self.optimizer, path,
            device=str(self.device), scheduler=self.scheduler,
        )
        self.epoch += 1  # start from next epoch

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------

    def _make_train_dataset(self):
        split = self.args.train_split  # 'train' or 'train_20s'
        if self.args.data_dir:
            return create_local_dataset(
                self.args.data_dir,
                batch_size=self.args.batch_size,
                shuffle_buffer=self.args.shuffle_buffer,
                split=split,
            )
        # Resolve GCS pattern: use explicit --gcs_train if provided,
        # otherwise look up by split name
        pattern = self.args.gcs_train
        if split != 'train' and pattern == GCS_TRAIN:
            # User chose a non-default split but didn't override --gcs_train
            pattern = GCS_SPLIT_PATTERNS.get(split, pattern)
        if self.args.subset_shards:
            # Restrict to first N shards by listing and re-joining
            import tensorflow as tf
            all_files = sorted(tf.io.gfile.glob(pattern))
            pattern = all_files[:self.args.subset_shards]
        return create_gcs_dataset(
            file_pattern=pattern,
            batch_size=self.args.batch_size,
            shuffle_buffer=self.args.shuffle_buffer,
        )

    def _make_val_dataset(self):
        split = self.args.val_split  # 'val' or 'val_interactive'
        if self.args.val_data_dir:
            return create_local_dataset(
                self.args.val_data_dir,
                batch_size=self.args.batch_size,
                shuffle_buffer=0,
                split=split,
            )
        # Resolve GCS pattern: use explicit --gcs_val if provided,
        # otherwise look up by split name
        pattern = self.args.gcs_val
        if split != 'val' and pattern == GCS_VAL:
            pattern = GCS_SPLIT_PATTERNS.get(split, pattern)
        return create_gcs_dataset(
            file_pattern=pattern,
            batch_size=self.args.batch_size,
            shuffle_buffer=0,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self, dataset) -> float:
        """
        Train for a portion of one epoch (yields steps per epoch batches).

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        accum   = self.args.accumulation_steps
        phase   = self.args.phase
        device  = self.device

        epoch_loss   = 0.0
        n_batches    = 0
        self.optimizer.zero_grad()

        for batch_idx, tf_batch in enumerate(dataset):
            torch_batch = tf_to_torch_batch(tf_batch, device=str(device))

            # Augmentation
            if self.args.augment:
                torch_batch = random_rotation_augmentation(torch_batch)
                torch_batch = random_translation_augmentation(torch_batch)

            # Forward + loss
            with autocast(enabled=self.scaler.is_enabled()):
                output     = self.model(torch_batch, phase=phase)
                loss_dict  = self.criterion(output, torch_batch, phase=phase)
                loss       = loss_dict['total'] / accum

            # Backward
            self.scaler.scale(loss).backward()

            # Accumulation step
            if (batch_idx + 1) % accum == 0:
                if self.args.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.args.log_every == 0:
                    self._log_step(loss_dict)

                # Step-level checkpoint
                if self.global_step % self.args.save_every == 0:
                    save_checkpoint(
                        self.model, self.optimizer,
                        self.epoch, self.global_step,
                        loss_dict['total'].item(),
                        self.args.checkpoint_dir,
                        scheduler=self.scheduler,
                    )

            epoch_loss += loss_dict['total'].item()
            n_batches  += 1

        return epoch_loss / max(n_batches, 1)

    def validate(self, val_dataset, max_val_batches: int = 100) -> float:
        """
        Run validation and return mean total loss.

        Args:
            val_dataset:      tf.data.Dataset for validation.
            max_val_batches:  Maximum number of batches to evaluate.

        Returns:
            val_loss: Mean validation loss.
        """
        self.model.eval()
        phase  = self.args.phase
        device = self.device
        total  = 0.0
        count  = 0

        with torch.no_grad():
            for tf_batch in val_dataset.take(max_val_batches):
                tb = tf_to_torch_batch(tf_batch, device=str(device))
                with autocast(enabled=self.scaler.is_enabled()):
                    out  = self.model(tb, phase=phase)
                    ld   = self.criterion(out, tb, phase=phase)
                total += ld['total'].item()
                count += 1

        val_loss = total / max(count, 1)
        logger.info(f'[Validation] step={self.global_step}  val_loss={val_loss:.4f}')

        if self.writer:
            self.writer.add_scalar('val/loss_total', val_loss, self.global_step)

        return val_loss

    def train(self):
        """Run the full training loop."""
        logger.info(f'Starting training (phase={self.args.phase}, '
                    f'epochs={self.args.num_epochs})')

        train_ds = self._make_train_dataset()
        val_ds   = self._make_val_dataset()

        steps_per_epoch = 1000  # approximate; tf.data with repeat doesn't expose len

        for epoch in range(self.epoch, self.args.num_epochs):
            self.epoch = epoch
            t0         = time.time()

            # Slice dataset into ~steps_per_epoch batches per epoch
            epoch_ds   = train_ds.take(steps_per_epoch)
            epoch_loss = self.train_epoch(epoch_ds)

            elapsed = time.time() - t0
            logger.info(f'Epoch {epoch:03d}/{self.args.num_epochs}  '
                        f'loss={epoch_loss:.4f}  '
                        f'time={elapsed:.1f}s  '
                        f'step={self.global_step}')

            if self.writer:
                self.writer.add_scalar('train/loss_epoch', epoch_loss, epoch)

            # Epoch checkpoint
            save_checkpoint(
                self.model, self.optimizer,
                epoch, self.global_step,
                epoch_loss, self.args.checkpoint_dir,
                scheduler=self.scheduler,
            )

            # Validation
            if (epoch + 1) % max(1, self.args.eval_every // steps_per_epoch) == 0:
                val_loss = self.validate(val_ds)
                save_best_checkpoint(
                    self.model, self.optimizer,
                    epoch, self.global_step,
                    val_loss, self.args.checkpoint_dir,
                    metric_name='val_loss',
                    higher_is_better=False,
                    scheduler=self.scheduler,
                )

            self.scheduler.step()

        logger.info('Training complete.')
        if self.writer:
            self.writer.close()

    # ------------------------------------------------------------------
    # Internal logging
    # ------------------------------------------------------------------

    def _log_step(self, loss_dict: dict):
        lr = self.optimizer.param_groups[0]['lr']
        msg = (f'step={self.global_step:08d}  '
               f"total={loss_dict['total'].item():.4f}  "
               f"goal={loss_dict['goal'].item():.4f}  "
               f"wta={loss_dict['wta'].item():.4f}  "
               f"phys={loss_dict['physics'].item():.4f}  "
               f"bound={loss_dict['boundary'].item():.4f}  "
               f"lr={lr:.2e}")
        logger.info(msg)

        if self.writer:
            for k, v in loss_dict.items():
                self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)
            self.writer.add_scalar('train/lr', lr, self.global_step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser  = build_arg_parser()
    args    = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
