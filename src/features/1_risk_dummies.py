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

    # Dummy risk (-1, 0, 1)
    df_max = df.groupby(ID)[risk].max().reset_index().rename(
        columns={risk: 'risk_max'})
    df_max['_dummy'+risk] = df_max['risk_max'].apply(
        lambda x: -1 if x < -15 else (0 if x < -10 else 1))

    df = pd.merge(df, df_max[[ID, '_dummy'+risk]], on=ID)

    # High x days risk
    def _high_xd_risk(df_group,
                      high_treshold: int = -6,
                      days: int = 2,
                      ) -> int:
        df_group['diff'] = abs(df_group[time]-days)
        risk2d = df_group[df_group['diff'] ==
                          df_group['diff'].min()][risk].values[0]
        return (1 if risk2d >= high_treshold else 0)

    df = pd.merge(df,
                  df.groupby(ID).apply(_high_xd_risk, days=0).to_frame(
                  ).reset_index().rename(columns={0: '_dummy_high_final_risk'}),
                  on=ID)

    df = pd.merge(df,
                  df.groupby(ID).apply(_high_xd_risk, days=1).to_frame(
                  ).reset_index().rename(columns={0: '_dummy_high_1d_risk'}),
                  on=ID)

    df = pd.merge(df,
                  df.groupby(ID).apply(_high_xd_risk, days=2).to_frame(
                  ).reset_index().rename(columns={0: '_dummy_high_2d_risk'}),
                  on=ID)

    return df
