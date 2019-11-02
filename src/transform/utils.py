# -*- coding: utf-8 -*-
import logging
import pandas as pd
import subprocess
from pathlib import Path
from typing import List, Callable


_UNIQUE_COLS = ["event_id","time_to_tca","mission_id","max_risk_estimate","max_risk_scaling",
    "miss_distance","relative_speed","relative_position_n","relative_position_r","relative_position_t",
    "relative_velocity_n","relative_velocity_r","relative_velocity_t","c_object_type","geocentric_latitude",
    "azimuth","elevation","F10","AP","F3M", "SSN"]


_PROCESSED_DATA_FOLDER = Path(__file__).resolve().parents[2].as_posix() + '/data/processed/'
_RAW_DATA_FOLDER = Path(__file__).resolve().parents[2].as_posix() + '/data/raw/'


def get_function_names(functions: List[Callable]) -> List[str]:
    return [f.__name__.split('_')[0] for f in functions]


def generate_output_name(base_name: str, transforms: List[str]) -> str:
    """ Generate name with current git commit hash and all transforms"""
    git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], 
                                        encoding='utf-8').strip()
    return f"{base_name}_{'-'.join(transforms)}_{git_commit}"


def save_transformed_dataset(base_name: str, transforms: List[Callable], df: pd.DataFrame) -> str:
    transforms_names = get_function_names(transforms)
    output_name = generate_output_name(base_name, transforms_names)
    df.to_csv(path_or_buf = _PROCESSED_DATA_FOLDER + output_name)


def read_csv(path: str, usecols: List[str] =_UNIQUE_COLS) -> pd.DataFrame:
    return pd.read_csv(_RAW_DATA_FOLDER + path, usecols=usecols)
