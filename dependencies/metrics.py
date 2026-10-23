from __future__ import annotations

import math
from typing import Dict

import numpy as np


def new_accumulator() -> Dict[str, float]:
    return {
        "count": 0, "phasor_sq": 0.0, "magnitude_sq": 0.0,
        "angle_abs": 0.0, "angle_count": 0, "max_phasor": 0.0,
    }


def update_metrics(acc: Dict[str, float], prediction: np.ndarray, truth: np.ndarray) -> None:
    prediction = np.asarray(prediction, np.complex128)
    truth = np.asarray(truth, np.complex128)
    difference = prediction - truth
    squared = np.square(np.abs(difference))
    magnitude_squared = np.square(np.abs(prediction) - np.abs(truth))
    valid = np.abs(truth) > 1e-6
    angle = np.abs(np.angle(prediction[valid] * np.conj(truth[valid]), deg=True))
    acc["count"] += truth.size
    acc["phasor_sq"] += float(squared.sum(dtype=np.float64))
    acc["magnitude_sq"] += float(magnitude_squared.sum(dtype=np.float64))
    acc["angle_abs"] += float(angle.sum(dtype=np.float64))
    acc["angle_count"] += angle.size
    acc["max_phasor"] = max(acc["max_phasor"], float(np.sqrt(squared.max())))


def finalize_metrics(acc: Dict[str, float]) -> Dict[str, float]:
    count = max(int(acc["count"]), 1)
    angle_count = max(int(acc["angle_count"]), 1)
    return {
        "complex_phasor_rmse_pu": math.sqrt(acc["phasor_sq"] / count),
        "ri_component_rmse_pu": math.sqrt(acc["phasor_sq"] / (2 * count)),
        "voltage_magnitude_rmse_pu": math.sqrt(acc["magnitude_sq"] / count),
        "voltage_angle_mae_deg": acc["angle_abs"] / angle_count,
        "max_complex_phasor_error_pu": acc["max_phasor"],
        "value_count": count,
    }
