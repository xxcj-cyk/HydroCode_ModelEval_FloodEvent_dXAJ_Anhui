import time
from pathlib import Path

import torch
from torch import optim

from hydromodels_dlm.config.model_config import PBM_PARAM_BOUNDS
from hydromodels_dlm.dataset.data_loader import FloodEventDataLoader
from hydromodels_dlm.dataset.data_source import group_file_ids_by_basin
from hydromodels_dlm.model.model_dict import MODEL_DICT
from hydromodels_dlm.model.physics_guided import (
    PhysicsGuidedModel,
    apply_cemaneige_climatology,
    apply_pbm_grad_steps,
    is_physics_model,
    make_physics_model,
    pbm_grad_steps,
)
from hydromodels_dlm.utils.device import resolve_device
from hydromodels_dlm.utils.logging import (
    get_logger,
    log_detail,
    log_epoch_progress,
    log_section,
    manual_progress_bar,
    progress_iter,
)
from hydromodels_dlm.utils.seed import set_random_seed
from hydromodels_dlm.workflow.artifacts import BEST_CHECKPOINT, NORMALIZATION_SCALER
from hydromodels_dlm.config.run_config import peak_focused_weights
from hydromodels_dlm.workflow.losses import LOSS_FUNCTIONS
from hydromodels_dlm.workflow.transfer import load_init_weights


MODEL_HYPERPARAM_KEYS = (
    'input_size',
    'output_size',
    'hidden_size',
    'dropout',
    'input_proj',
)


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------


def read_model_hyperparam(cfg):
    return dict((cfg.get('model_cfgs') or {}).get('model_hyperparam') or {})


def expected_output_size(cfg):
    name = str(cfg['model_cfgs']['model_name'])
    if is_physics_model(name):
        return len(PBM_PARAM_BOUNDS[name])
    outputs = (cfg.get('data_cfgs') or {}).get('variables', {}).get('dynamic_outputs') or {}
    if not outputs:
        raise ValueError('data_cfgs.variables.dynamic_outputs required to validate output_size')
    return len(outputs)


def validate_model_hyperparam(cfg, n_inputs):
    hp = {k: v for k, v in read_model_hyperparam(cfg).items() if k in MODEL_HYPERPARAM_KEYS}
    unknown = set(hp) - set(MODEL_HYPERPARAM_KEYS)
    if unknown:
        raise ValueError(
            f'unknown model_hyperparam keys {sorted(unknown)!r}; allowed: {MODEL_HYPERPARAM_KEYS}'
        )

    for key in ('input_size', 'output_size'):
        if key not in hp or hp[key] is None:
            raise ValueError(
                f'model_hyperparam.{key} must be set explicitly in config (no default)'
            )

    input_size = int(hp['input_size'])
    output_size = int(hp['output_size'])
    if input_size != int(n_inputs):
        raise ValueError(
            f'model_hyperparam.input_size={input_size} != data features {n_inputs}'
        )

    expected_out = expected_output_size(cfg)
    if output_size != expected_out:
        raise ValueError(
            f'model_hyperparam.output_size={output_size} != expected {expected_out} '
            f"for model {cfg['model_cfgs']['model_name']!r}"
        )

    return {
        'input_size': input_size,
        'output_size': output_size,
        'hidden_size': hp.get('hidden_size', 32),
        'dropout': hp.get('dropout', 0.0),
        'input_proj': hp.get('input_proj'),
    }


def build_model(cfg, *, n_inputs, warmup_length=365):
    name = cfg['model_cfgs']['model_name']
    entry = MODEL_DICT.get(name)
    if entry is None:
        raise ValueError(f'unknown model {name!r}; registered: {sorted(MODEL_DICT)}')

    hp = validate_model_hyperparam(cfg, n_inputs)
    if is_physics_model(name):
        pb_core = entry(warmup_length=warmup_length)
        return make_physics_model(
            n_lstm_inputs=hp['input_size'],
            output_size=hp['output_size'],
            hidden_size=hp['hidden_size'],
            dropout=hp['dropout'],
            input_proj=hp['input_proj'],
            pb_core=pb_core,
        )

    return entry(
        input_size=hp['input_size'],
        output_size=hp['output_size'],
        hidden_size=hp['hidden_size'],
        dropout=hp['dropout'],
        input_proj=hp['input_proj'],
    )


# ---------------------------------------------------------------------------
# Optimizer and criterion
# ---------------------------------------------------------------------------


def epoch_learning_rate(base_lr, decay, epoch):
    return float(base_lr) * (float(decay) ** (int(epoch) - 1))


def set_optimizer_learning_rate(optimizer, lr):
    value = float(lr)
    for group in optimizer.param_groups:
        group['lr'] = value
    return value


def build_optimizer(model, cfg):
    name = str(cfg['training_cfgs']['optimizer_name'])
    if name != 'Adam':
        raise ValueError(f'unsupported optimizer {name!r}')
    return optim.Adam(model.parameters(), lr=float(cfg['training_cfgs']['learning_rate']))


def build_criterion(cfg):
    name = str(cfg['training_cfgs']['loss_function']).upper()
    if name == 'PEAKFOCUSED':
        return LOSS_FUNCTIONS[name](**peak_focused_weights(cfg))
    return LOSS_FUNCTIONS[name]()


# ---------------------------------------------------------------------------
# Batch forward
# ---------------------------------------------------------------------------


def model_forward(model, batch, warmup):
    if isinstance(model, PhysicsGuidedModel):
        drivers, lstm_inputs, _y, _basin_idx = batch
        return model(drivers, lstm_inputs)
    x, _y = batch
    return model(x)[:, warmup:, :]


def forward_batch(model, batch, device, warmup, *, physics=False):
    if physics:
        drivers_b, lstm_b, yb, basin_idx = batch
        drivers_b = drivers_b.to(device, non_blocking=True)
        lstm_b = lstm_b.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        basin_idx = basin_idx.to(device, non_blocking=True)
        drivers = drivers_b.transpose(0, 1)
        pred = model_forward(
            model,
            (drivers, lstm_b, yb, basin_idx),
            warmup,
        ).transpose(0, 1)
        return pred, yb

    xb, yb = batch
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    pred = model_forward(model, (xb, yb), warmup)
    return pred, yb


def optimizer_step(loss, model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------


def format_rate(value):
    return f'{float(value):g}'


def log_gpu_memory(log, device, label):
    if device.type != 'cuda':
        return
    alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    log_detail(
        log,
        'GPU memory [%s]: peak allocated %.2f GiB, reserved %.2f GiB',
        label,
        alloc,
        reserved,
    )


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    warmup,
    *,
    desc='Train',
    show_bar=True,
    bar_leave=True,
    physics=False,
):
    model.train()
    total = 0.0
    n_batches = 0
    batch_iter = progress_iter(
        loader,
        desc=desc,
        enabled=show_bar,
        unit='batch',
        leave=bar_leave,
    )
    for batch in batch_iter:
        pred, yb = forward_batch(model, batch, device, warmup, physics=physics)
        loss = criterion(pred, yb)
        optimizer_step(loss, model, optimizer)
        total += float(loss.detach())
        n_batches += 1
    return total / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------


def logical_basin_ids(loader, basin_ids):
    if isinstance(loader, FloodEventDataLoader):
        return sorted(group_file_ids_by_basin(basin_ids).keys())
    return sorted(str(b) for b in basin_ids)


def count_sliding_valid_batches(
    loader,
    basin_ids,
    scalers,
    *,
    period,
    batch_size,
    num_workers,
    model_name=None,
):
    period_key = str(period).lower()
    total = 0
    for basin_id in logical_basin_ids(loader, basin_ids):
        valid_loader = loader.build_period_loader(
            [basin_id],
            scalers,
            period=period_key,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            drop_last=False,
            model_name=model_name,
        )
        total += len(valid_loader)
    return total


def basin_sliding_valid_loss(
    model,
    loader,
    basin_id,
    period,
    scalers,
    criterion,
    device,
    *,
    batch_size,
    num_workers=0,
    model_name=None,
    physics=False,
    progress=None,
):
    period_key = str(period).lower()
    sample_warmup = 0 if physics else int(loader.warmup_length)
    valid_loader = loader.build_period_loader(
        [basin_id],
        scalers,
        period=period_key,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
        model_name=model_name,
        pin_memory=device.type == 'cuda',
    )
    if len(valid_loader.dataset) == 0:
        raise ValueError(
            f'validation sampler has zero samples for basin {basin_id!r} '
            f'period {period_key!r}'
        )

    pred_parts = []
    target_parts = []
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            pred, yb = forward_batch(
                model,
                batch,
                device,
                sample_warmup,
                physics=physics,
            )
            pred_parts.append(pred.reshape(-1, pred.shape[-1]))
            target_parts.append(yb.reshape(-1, yb.shape[-1]))
    pred_all = torch.cat(pred_parts, dim=0)
    target_all = torch.cat(target_parts, dim=0)
    if progress is not None:
        progress.update(len(valid_loader))
    return float(criterion(pred_all, target_all).detach())


def basin_equal_sliding_valid_loss(
    model,
    loader,
    basin_ids,
    period,
    scalers,
    criterion,
    device,
    *,
    batch_size,
    num_workers=0,
    model_name=None,
    physics=False,
    show_bar=False,
    desc='Valid',
    bar_leave=False,
):
    basin_list = logical_basin_ids(loader, basin_ids)
    progress = None
    if show_bar:
        progress = manual_progress_bar(
            total=count_sliding_valid_batches(
                loader,
                basin_ids,
                scalers,
                period=period,
                batch_size=batch_size,
                num_workers=num_workers,
                model_name=model_name,
            ),
            desc=desc,
            unit='batch',
            leave=bar_leave,
        )
    try:
        losses = [
            basin_sliding_valid_loss(
                model,
                loader,
                basin_id,
                period,
                scalers,
                criterion,
                device,
                batch_size=batch_size,
                num_workers=num_workers,
                model_name=model_name,
                physics=physics,
                progress=progress,
            )
            for basin_id in basin_list
        ]
    finally:
        if progress is not None:
            progress.close()
    if not losses:
        raise ValueError(f'no validation samples for period {period!r}')
    return sum(losses) / len(losses)


def valid_loss(
    model,
    loader,
    basin_ids,
    period,
    scalers,
    criterion,
    device,
    *,
    batch_size,
    num_workers=0,
    desc='Valid',
    show_bar=True,
    bar_leave=True,
    model_name=None,
):
    physics = bool(model_name and is_physics_model(model_name))
    return basin_equal_sliding_valid_loss(
        model,
        loader,
        basin_ids,
        period,
        scalers,
        criterion,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=model_name,
        physics=physics,
        show_bar=show_bar,
        desc=desc,
        bar_leave=bar_leave,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def log_training_setup(log, cfg, train_sampler, physics):
    tc = cfg['training_cfgs']
    warmup = int(train_sampler.warmup)
    horizon = int(train_sampler.horizon)
    base_lr = float(tc['learning_rate'])
    lr_decay = float(tc.get('learning_rate_decay', 1.0))
    log_section(log, '▶ Training')
    loss_name = tc.get('loss_function', 'RMSE')
    log_detail(
        log,
        'Optimizer: %s  |  Loss: %s  |  Learning rate: %s (decay: %s)',
        tc.get('optimizer_name', 'Adam'),
        loss_name,
        format_rate(base_lr),
        format_rate(lr_decay),
    )
    if str(loss_name).upper() == 'PEAKFOCUSED':
        weights = peak_focused_weights(cfg)
        log_detail(
            log,
            'PeakFocused weights: overall=%s  peak=%s  high=%s',
            format_rate(weights['overall_weight']),
            format_rate(weights['peak_weight']),
            format_rate(weights['high_weight']),
        )
    log_detail(
        log,
        'Batch_Size: %d  |  Train samples: %d  |  sequence=warmup+forecast: %d+%d  |',
        int(tc['batch_size']),
        len(train_sampler),
        warmup,
        horizon,
    )
    log_detail(
        log,
        'Epochs: %d  |  Patience: %d',
        int(tc['epochs']),
        int(tc['patience']),
    )
    if physics:
        steps = pbm_grad_steps(cfg, horizon)
        log_detail(
            log,
            'PBM grad_steps: %s (forecast_length=%d)',
            'full' if steps <= 0 else steps,
            int(horizon),
        )


def train_model(
    cfg,
    loader,
    basin_ids,
    out_dir,
    device=None,
    *,
    init_weight_path=None,
    scalers=None,
):
    log = get_logger('train')
    tc = cfg['training_cfgs']
    set_random_seed(tc['random_seed'])
    device = device or resolve_device(tc.get('device'))
    model_name = cfg['model_cfgs']['model_name']
    physics = is_physics_model(model_name)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if scalers is None:
        scalers = loader.fit_scalers(basin_ids)
    scaler_path = out_dir / NORMALIZATION_SCALER
    loader.save_scalers(scaler_path, scalers)

    train_sampler = loader.build_train_sampler(
        basin_ids, scalers, model_name=model_name
    )
    if len(train_sampler) == 0:
        raise ValueError('training sampler has zero samples')

    batch_size = int(tc['batch_size'])
    num_workers = int(tc.get('num_workers', 0))
    pin_memory = device.type == 'cuda'
    train_loader = loader.build_train_loader(
        basin_ids,
        scalers,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(train_sampler) > batch_size,
        model_name=model_name,
        pin_memory=pin_memory,
    )

    n_inputs = loader.model_input_size(scalers)
    model = build_model(
        cfg,
        n_inputs=n_inputs,
        warmup_length=loader.warmup_length,
    ).to(device)
    if physics:
        apply_pbm_grad_steps(model, cfg, loader.forecast_length)
        apply_cemaneige_climatology(model, loader, basin_ids)
    if init_weight_path:
        load_init_weights(model, init_weight_path, loader, scalers, device)
        log_detail(log, 'Init weights: %s', init_weight_path)
    optimizer = build_optimizer(model, cfg)
    criterion = build_criterion(cfg)
    sample_warmup = 0 if physics else int(loader.warmup_length)

    epochs = int(tc['epochs'])
    patience = int(tc['patience'])
    best_loss = float('inf')
    best_epoch = 0
    stale = 0

    log_training_setup(log, cfg, train_sampler, physics)

    for epoch in range(1, epochs + 1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        lr = set_optimizer_learning_rate(
            optimizer,
            epoch_learning_rate(
                float(tc['learning_rate']),
                float(tc.get('learning_rate_decay', 1.0)),
                epoch,
            ),
        )
        t0 = time.perf_counter()
        tr_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            sample_warmup,
            desc=f'Epoch {epoch}/{epochs} Train',
            show_bar=True,
            bar_leave=False,
            physics=physics,
        )
        log_epoch_progress(
            log, epoch, epochs, 'train', len(train_loader),
            time.perf_counter() - t0,
        )
        if epoch == 1:
            log_gpu_memory(log, device, 'after epoch 1 train')
        va_steps = count_sliding_valid_batches(
            loader,
            basin_ids,
            scalers,
            period='valid',
            batch_size=batch_size,
            num_workers=num_workers,
            model_name=model_name,
        )
        t0 = time.perf_counter()
        va_loss = valid_loss(
            model,
            loader,
            basin_ids,
            'valid',
            scalers,
            criterion,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            desc=f'Epoch {epoch}/{epochs} Valid',
            show_bar=True,
            bar_leave=False,
            model_name=model_name,
        )
        log_epoch_progress(
            log, epoch, epochs, 'valid', va_steps,
            time.perf_counter() - t0,
        )
        improved = va_loss < best_loss
        log_detail(
            log,
            'Epoch %d/%d Lr=%s  |  Train Loss=%.4f  |  Valid Loss=%.4f',
            epoch,
            epochs,
            format_rate(lr),
            tr_loss,
            va_loss,
        )

        if improved:
            stale = 0
            best_epoch = epoch
            best_loss = va_loss
            ckpt = {
                'model_state': model.state_dict(),
                'model_name': model_name,
                'n_inputs': n_inputs,
                'model_hyperparam': validate_model_hyperparam(cfg, n_inputs),
            }
            torch.save(ckpt, out_dir / BEST_CHECKPOINT)
            log_detail(log, 'Model Update')
        else:
            stale += 1
            log_detail(log, 'Epochs without Model Update: %d/%d', stale, patience)
            if stale >= patience:
                log_detail(log, 'Early stop at epoch %d', epoch)
                break

    return {
        'out_dir': str(out_dir),
        'best_valid_loss': best_loss,
        'best_epoch': best_epoch,
        'epochs_run': epoch,
        'scaler_path': str(scaler_path),
    }
