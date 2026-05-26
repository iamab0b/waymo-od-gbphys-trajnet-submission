"""
Checkpoint utilities for GB-Phys TrajNet training.

Handles saving / loading model state, finding the latest checkpoint,
and tracking the best checkpoint by a validation metric.
"""

import os
import glob
import re
import torch


def save_checkpoint(model, optimizer, epoch: int, step: int, loss: float,
                    checkpoint_dir: str, scheduler=None,
                    extra_info: dict = None) -> str:
    """
    Save a training checkpoint to disk.

    Args:
        model:          PyTorch nn.Module.
        optimizer:      PyTorch optimizer.
        epoch:          Current epoch (0-indexed).
        step:           Current global training step.
        loss:           Current loss value.
        checkpoint_dir: Directory to save checkpoints in.
        scheduler:      Optional learning-rate scheduler.
        extra_info:     Optional dict of additional metadata to save.

    Returns:
        save_path: Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename  = f'ckpt_epoch{epoch:04d}_step{step:08d}.pt'
    save_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        'epoch':       epoch,
        'step':        step,
        'loss':        loss,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    if extra_info is not None:
        checkpoint.update(extra_info)

    torch.save(checkpoint, save_path)
    print(f'[Checkpoint] Saved: {save_path}  (epoch={epoch}, step={step}, loss={loss:.4f})')
    return save_path


def load_checkpoint(model, optimizer, path: str, device: str = 'cpu',
                    scheduler=None) -> tuple:
    """
    Load a checkpoint from disk into model and optimizer.

    Args:
        model:     PyTorch nn.Module.
        optimizer: PyTorch optimizer (will be updated in-place).
        path:      Path to checkpoint file.
        device:    Device string ('cpu', 'cuda', etc.).
        scheduler: Optional scheduler (will be updated in-place if provided).

    Returns:
        (epoch, step, loss)  values from the checkpoint.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state'])

    # Only restore optimizer state if the parameter groups match.
    # Between training phases the optimizer covers a different set of
    # parameters (e.g. Phase 1 freezes Stage 2, Phase 2 freezes Stage 1),
    # so the saved state is incompatible with the new optimizer.  In that
    # case we silently skip optimizer restoration and start fresh.
    try:
        optimizer.load_state_dict(checkpoint['optim_state'])
    except (ValueError, KeyError) as e:
        print(f'[Checkpoint] Skipping optimizer state restore '
              f'(parameter groups changed between phases): {e}')

    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    epoch = checkpoint.get('epoch', 0)
    step  = checkpoint.get('step', 0)
    loss  = checkpoint.get('loss', float('inf'))

    print(f'[Checkpoint] Loaded: {path}  (epoch={epoch}, step={step}, loss={loss:.4f})')
    return epoch, step, loss


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Scan a directory for checkpoints and return the path with the highest step.

    Looks for files matching the pattern `ckpt_epoch*_step*.pt`.

    Args:
        checkpoint_dir: Directory to scan.

    Returns:
        Path to the latest checkpoint file, or None if none found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, 'ckpt_epoch*_step*.pt')
    candidates = glob.glob(pattern)

    if not candidates:
        return None

    def _extract_step(path: str) -> int:
        m = re.search(r'step(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else -1

    latest = max(candidates, key=_extract_step)
    return latest


def save_best_checkpoint(model, optimizer, epoch: int, step: int,
                         metric_value: float, checkpoint_dir: str,
                         metric_name: str = 'soft_map',
                         higher_is_better: bool = True,
                         scheduler=None) -> bool:
    """
    Save a checkpoint only if it represents the best observed metric.

    Maintains a `best_{metric_name}.pt` file in checkpoint_dir.

    Args:
        model:            PyTorch nn.Module.
        optimizer:        PyTorch optimizer.
        epoch:            Current epoch.
        step:             Current global step.
        metric_value:     Value of the evaluation metric.
        checkpoint_dir:   Directory to save checkpoints.
        metric_name:      Name of the metric (used in filename).
        higher_is_better: If True, higher metric is better.
        scheduler:        Optional scheduler.

    Returns:
        True if this was a new best and the checkpoint was saved.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, f'best_{metric_name}.pt')
    tracker   = os.path.join(checkpoint_dir, f'.best_{metric_name}.txt')

    # Read previous best
    prev_best = None
    if os.path.isfile(tracker):
        with open(tracker, 'r') as f:
            try:
                prev_best = float(f.read().strip())
            except ValueError:
                prev_best = None

    # Compare
    is_new_best = (prev_best is None or
                   (higher_is_better and metric_value > prev_best) or
                   (not higher_is_better and metric_value < prev_best))

    if is_new_best:
        checkpoint = {
            'epoch':        epoch,
            'step':         step,
            'loss':         metric_value,
            'metric_name':  metric_name,
            'metric_value': metric_value,
            'model_state':  model.state_dict(),
            'optim_state':  optimizer.state_dict(),
        }
        if scheduler is not None:
            checkpoint['scheduler_state'] = scheduler.state_dict()

        torch.save(checkpoint, best_path)

        with open(tracker, 'w') as f:
            f.write(str(metric_value))

        print(f'[Checkpoint] New best {metric_name}={metric_value:.4f}  '
              f'(prev={prev_best})  -> {best_path}')

    return is_new_best
