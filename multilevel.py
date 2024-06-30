from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from prophet import Prophet
from sklearn.linear_model import Ridge

def aggregate_forecasts(forecasts_df, val_start_date_str):
    # forecasts df will contain several forecasts at different levels
    train_forecasts_df = forecasts_df[forecasts_df["ds"] < val_start_date_str]
    val_forecasts_df = forecasts_df[forecasts_df["ds"] >= val_start_date_str]
    covariate_colnames = ["yhat_1", "yhat_2", "yhat_3"]


    def fit_agg_linreg(x_df):
        y = x_df["y"].to_numpy()
        X = x_df[covariate_colnames].to_numpy()

        model = Ridge(alpha=1e-2)
        model.fit(X, y)
        df_dict = {}
        df_dict["intercept"] = [model.coef_[0]]
        for i, cn in enumerate(covariate_colnames):
            df_dict[cn] = [model.coef_[i + 1]]
        return pd.DataFrame(df_dict)

    agg_coefs_df = train_forecasts_df.groupby(["cluster", "store_nbr", "family"]).apply(fit_agg_linreg).reset_index()

    def pred_agg_linreg(x_df):
        # query the correct linear model and then aggregate the input cell (x_df)
        pass
    pass