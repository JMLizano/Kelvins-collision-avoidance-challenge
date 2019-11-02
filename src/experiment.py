""" 
Description of the experiment and objectives
"""
import pandas as pd
import data.utils as utils
from typing import List, Callable

_UNIQUE_COLS = ["event_id","time_to_tca","mission_id","max_risk_estimate","max_risk_scaling",
    "miss_distance","relative_speed","relative_position_n","relative_position_r","relative_position_t",
    "relative_velocity_n","relative_velocity_r","relative_velocity_t","c_object_type","geocentric_latitude",
    "azimuth","elevation","F10","AP","F3M", "SSN"]


class Experiment:
    
    def __init__(self, id:str):
        self.id = id
    
    def load_data(self):
        return utils.read_csv(self.raw_data)

    def apply(self,functions: List[Callable], df: pd.DataFrame) -> pd.DataFrame:
        for f in functions:
            df = f(df)
        return df
    
    def apply_transforms(self, df: pd.DataFrame):
        self.apply(self.transforms, df)
        utils.save_transformed_dataset(base_name=self.id, transforms=self.transforms, df=df)
    
    def run(self):
        df = self.load_data()
        df = self.apply_transforms(df)