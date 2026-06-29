import numpy as np
import torch
from torch.utils.data import Dataset

from hydromodels_dlm.dataset.data_loader import (
    event_flood_target_mask,
    get_input_layout,
    norm_static_vector,
    prepare_period_basin,
    prepare_train_basin,
    train_window_lookup,
    train_window_slice,
)
from hydromodels_dlm.dataset.input_layout import merge_static_numpy


# ---------------------------------------------------------------------------
# Event packs
# ---------------------------------------------------------------------------


def build_period_event_packs(
    loader,
    basin_ids,
    scalers,
    layout,
    period,
    *,
    driver_indices=None,
):
    packs = []
    for bid in basin_ids:
        hub = scalers[bid]
        static = (
            norm_static_vector(loader, hub, bid)
            if layout['uses_static']
            else None
        )
        for bundle in loader.iter_period_bundles(bid, period):
            x_raw, y_raw, _ = loader.bundle_arrays(bundle)
            x_norm = hub.x_scaler.transform(x_raw)
            pack = {
                'basin_id': bid,
                'x_norm': x_norm,
                'x': merge_static_numpy(x_norm, static, layout),
                'y_norm': hub.y_scaler.transform(y_raw),
                'y_raw': np.asarray(y_raw, dtype=np.float64),
                'flood_target': event_flood_target_mask(bundle['series']),
            }
            if driver_indices is not None:
                pack['drivers'] = x_raw[:, driver_indices]
            packs.append(pack)
    return packs


# ---------------------------------------------------------------------------
# Long-term sliding window
# ---------------------------------------------------------------------------


class SlidingWindowTrainSampler(Dataset):

    def __init__(self, loader, basin_ids, scalers, input_layout=None, *, period='train'):
        self.basin_ids = [str(b) for b in basin_ids]
        self.period = str(period).lower()
        self.layout = get_input_layout(input_layout or loader.input_layout_name)
        self.warmup = int(loader.warmup_length)
        self.horizon = int(loader.forecast_length)

        self.x_norm = []
        self.x = []
        self.y = []
        for bid in self.basin_ids:
            pack = prepare_period_basin(
                loader, bid, self.period, scalers[bid], self.layout
            )
            self.x_norm.append(pack['x_norm'])
            self.x.append(merge_static_numpy(pack['x_norm'], pack['static'], self.layout))
            self.y.append(pack['y_norm'])

        self.lookup = train_window_lookup(
            self.x_norm,
            self.y,
            warmup=self.warmup,
            horizon=self.horizon,
        )

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, index):
        basin_idx, start = self.lookup[index]
        input_sl, target_sl = train_window_slice(start, self.warmup, self.horizon)
        return (
            torch.from_numpy(self.x[basin_idx][input_sl].astype(np.float32)),
            torch.from_numpy(self.y[basin_idx][target_sl].astype(np.float32)),
        )


# ---------------------------------------------------------------------------
# Long-term physics-guided
# ---------------------------------------------------------------------------


class PhysicsGuidedTrainSampler(Dataset):

    def __init__(self, loader, basin_ids, scalers, *, driver_indices, period='train'):
        self.basin_ids = [str(b) for b in basin_ids]
        self.period = str(period).lower()
        self.layout = get_input_layout(loader.input_layout_name)
        self.warmup = int(loader.warmup_length)
        self.horizon = int(loader.forecast_length)

        self.drivers = []
        self.x_norm = []
        self.lstm_x = []
        self.y_norm = []
        self.y_raw = []
        for bid in self.basin_ids:
            pack = prepare_period_basin(
                loader,
                bid,
                self.period,
                scalers[bid],
                self.layout,
                driver_indices=driver_indices,
            )
            self.drivers.append(pack['drivers'])
            self.x_norm.append(pack['x_norm'])
            self.lstm_x.append(merge_static_numpy(pack['x_norm'], pack['static'], self.layout))
            self.y_norm.append(pack['y_norm'])
            self.y_raw.append(pack['y_raw'])

        self.lookup = train_window_lookup(
            self.x_norm,
            self.y_norm,
            warmup=self.warmup,
            horizon=self.horizon,
        )

    def __len__(self):
        return len(self.lookup)

    @property
    def y(self):
        return self.y_raw

    def __getitem__(self, index):
        basin_idx, start = self.lookup[index]
        input_sl, target_sl = train_window_slice(start, self.warmup, self.horizon)
        return (
            torch.from_numpy(self.drivers[basin_idx][input_sl].astype(np.float32)),
            torch.from_numpy(self.lstm_x[basin_idx][input_sl].astype(np.float32)),
            torch.from_numpy(self.y_raw[basin_idx][target_sl].astype(np.float32)),
            basin_idx,
        )


def physics_guided_collate(batch):
    return (
        torch.stack([item[0] for item in batch]),
        torch.stack([item[1] for item in batch]),
        torch.stack([item[2] for item in batch]),
        torch.tensor([item[3] for item in batch], dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Flood-event sliding window
# ---------------------------------------------------------------------------


class FloodEventSlidingWindowTrainSampler(Dataset):

    def __init__(self, loader, basin_ids, scalers, input_layout=None, *, period='train'):
        self.basin_ids = [str(b) for b in basin_ids]
        self.period = str(period).lower()
        self.layout = get_input_layout(input_layout or loader.input_layout_name)
        self.warmup = int(loader.warmup_length)
        self.horizon = int(loader.forecast_length)

        self.x_norm = []
        self.x = []
        self.y = []
        self.flood_target = []
        for pack in build_period_event_packs(
            loader,
            self.basin_ids,
            scalers,
            self.layout,
            self.period,
        ):
            self.x_norm.append(pack['x_norm'])
            self.x.append(pack['x'])
            self.y.append(pack['y_norm'])
            self.flood_target.append(pack['flood_target'])

        self.lookup = train_window_lookup(
            self.x_norm,
            self.y,
            warmup=self.warmup,
            horizon=self.horizon,
            flood_target_mask=self.flood_target,
        )

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, index):
        event_idx, start = self.lookup[index]
        input_sl, target_sl = train_window_slice(start, self.warmup, self.horizon)
        return (
            torch.from_numpy(self.x[event_idx][input_sl].astype(np.float32)),
            torch.from_numpy(self.y[event_idx][target_sl].astype(np.float32)),
        )


class FloodEventPhysicsGuidedTrainSampler(Dataset):

    def __init__(self, loader, basin_ids, scalers, *, driver_indices, period='train'):
        self.basin_ids = [str(b) for b in basin_ids]
        self.period = str(period).lower()
        self.layout = get_input_layout(loader.input_layout_name)
        self.warmup = int(loader.warmup_length)
        self.horizon = int(loader.forecast_length)

        self.drivers = []
        self.x_norm = []
        self.lstm_x = []
        self.y_norm = []
        self.y_raw = []
        self.flood_target = []
        self.basin_idx = []
        basin_id_map = {bid: idx for idx, bid in enumerate(self.basin_ids)}

        for pack in build_period_event_packs(
            loader,
            self.basin_ids,
            scalers,
            self.layout,
            self.period,
            driver_indices=driver_indices,
        ):
            self.drivers.append(pack['drivers'])
            self.x_norm.append(pack['x_norm'])
            self.lstm_x.append(pack['x'])
            self.y_norm.append(pack['y_norm'])
            self.y_raw.append(pack['y_raw'])
            self.flood_target.append(pack['flood_target'])
            self.basin_idx.append(basin_id_map[pack['basin_id']])

        self.lookup = train_window_lookup(
            self.x_norm,
            self.y_norm,
            warmup=self.warmup,
            horizon=self.horizon,
            flood_target_mask=self.flood_target,
        )

    def __len__(self):
        return len(self.lookup)

    @property
    def y(self):
        return self.y_raw

    def __getitem__(self, index):
        event_idx, start = self.lookup[index]
        input_sl, target_sl = train_window_slice(start, self.warmup, self.horizon)
        return (
            torch.from_numpy(self.drivers[event_idx][input_sl].astype(np.float32)),
            torch.from_numpy(self.lstm_x[event_idx][input_sl].astype(np.float32)),
            torch.from_numpy(self.y_raw[event_idx][target_sl].astype(np.float32)),
            self.basin_idx[event_idx],
        )
