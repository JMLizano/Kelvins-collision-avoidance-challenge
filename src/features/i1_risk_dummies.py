import pandas as pd
import numpy as np
import skfda
import os
import gc


def create_risk_dummies(df: pd.DataFrame,
                        ID: str = 'event_id',
                        time: str = 'time_to_tca',
                        risk: str = 'risk'
                        ) -> pd.DataFrame:
    """
    Creates dummy variables from the risk:
    - _dummy_max_risk: -1, 0 or 1 depending on wheter the maximum risk
      is less than -15 or -10 respectively.
    - _dummy_high_final_risk: 1 if the risk for the last CDM row is 
      higher than -6
    - _dummy_high_1d_risk: 1 if the risk one day before the tca is 
      higher than -6  
    - _dummy_high_2d_risk: 1 if the risk two days before the tca is 
      higher than -6  
    """

    # Dummy risk (-1, 0, 1)
    print('Creating dummy for maximum risk')
    df_max = df.groupby(ID)[risk].max().reset_index().rename(
        columns={risk: 'risk_max'})
    df_max['_dummy_max_'+risk] = df_max['risk_max'].apply(
        lambda x: -1 if x < -15 else (0 if x < -10 else 1))

    df = pd.merge(df, df_max[[ID, '_dummy_max_'+risk]], on=ID)

    # High x days risk
    def _high_xd_risk(df_group,
                      high_treshold: int = -6,
                      days: int = 2,
                      ) -> int:
        df_group['diff'] = abs(df_group[time]-days)
        risk2d = df_group[df_group['diff'] ==
                          df_group['diff'].min()][risk].values[0]
        return (1 if risk2d >= high_treshold else 0)

    print('Creating high final risk')
    df = pd.merge(df,
                  df.groupby(ID).apply(_high_xd_risk, days=0).to_frame(
                  ).reset_index().rename(columns={0: '_dummy_high_final_risk'}),
                  on=ID)

    print('Creating high risk 1 d before tca')
    df = pd.merge(df,
                  df.groupby(ID).apply(_high_xd_risk, days=1).to_frame(
                  ).reset_index().rename(columns={0: '_dummy_high_1d_risk'}),
                  on=ID)

    print('Creating high risk 2 d before tca')
    df = pd.merge(df,
                  df.groupby(ID).apply(_high_xd_risk, days=2).to_frame(
                  ).reset_index().rename(columns={0: '_dummy_high_2d_risk'}),
                  on=ID)

    return df
