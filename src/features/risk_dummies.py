import pandas as pd
import numpy as np
import skfda
import os
import gc


def f0_create_risk_dummies(df: pd.DataFrame,
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

    df['day'] = df[time].round().astype(int)

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
        df_riskxd = df_group[(df_group['diff'] == df_group['diff'].min()) & (
            df_group['day'] <= days)]

        if len(df_riskxd) > 0:
            riskxd = df_riskxd[risk].values[0]
        else:
            riskxd = np.nan

        return (1 if riskxd >= high_treshold else 0)

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


def f1_create_manoeuvre_dummies(df_original: pd.DataFrame,
                                ID: str = 'event_id',
                                time: str = 'time_to_tca',
                                risk: str = 'risk',
                                jump: int = -24
                                ) -> pd.DataFrame:
    """
    Create a dummy variable, _dummy_manoeuvre, to mark the events for 
    which a manoeuvre was made in one of the previous days. It is 
    defined as the events for which the difference between consecutive 
    records on the last 2 days is greater than -24 (by default) 
    """
    df = df_original.copy()
    df['day'] = df[time].round().astype(int)

    df2d = df[df['day'] <= 2].copy()
    df2d['risk_diff'] = df2d.groupby(ID)[risk].diff().fillna(0)

    df2d['jump'] = (df2d['risk_diff'] <= jump)*1
    df2d = pd.merge(df,
                    df2d.groupby(ID)['jump'].max().to_frame().rename(
                        columns={'jump': '_dummy_manoeuvre'}),
                    on=ID)

    return df2d


def f2_create_high_risk(df_original: pd.DataFrame
                        ) -> pd.DataFrame:
    """
    We define our high-risk class as the event which satisfies one 
    of the two conditions:
      - to have high risk in the last day or,
      - a manoeuvre was detected in the last 2 days
    """
    df = df_original.copy()

    if '_dummy_high_final_risk' not in df.columns:
        df = create_risk_dummies(df)

    if '_dummy_manoeuvre 'not in df.columns:
        df = create_manoeuvre_dummies(df)

    df['_high_risk'] = (df['_dummy_high_final_risk'] +
                        df['_dummy_manoeuvre']).apply(lambda x: min(x, 1)).values
    return df
