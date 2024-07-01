from datetime import date, datetime, timedelta
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from prophet import Prophet
from sklearn.linear_model import Ridge

def msle(preds_df):
    return np.mean((np.log(1 + preds_df["y"].values) - np.log(1 + preds_df["yhat"].values))**2)

def cross_validation(train_df, key_cols_list, hp_df_list, stores_dict, all_holidays_df, train_days=365*4, val_days=16, interval=64):
    all_key_cols = ["cluster", "store_nbr", "family"]
    # get important dates
    d0 = datetime.strptime(train_df["ds"].iloc[0], "%Y-%m-%d").date()
    d_max = datetime.strptime(train_df["ds"].iloc[-1], "%Y-%m-%d").date()

    def fit_prophet(x_df, d1, d2, d3):
        # retrieve h_df
        store_nbrs = x_df["store_nbr"].drop_duplicates()
        states = [stores_dict[snbr]["state"] for snbr in store_nbrs]
        cities = [stores_dict[snbr]["city"] for snbr in store_nbrs]
        filter = (all_holidays_df["locale_name"] == "Ecuador")
        for s in states:
            filter = filter | (all_holidays_df["locale_name"] == s)
        for c in cities:
            filter = filter | (all_holidays_df["locale_name"] == c)
        h_df = all_holidays_df[filter]
        h_df = h_df[["ds", "holiday", "lower_window", "upper_window"]]

        train_filter = (x_df["ds"] >= d1) & (x_df["ds"] < d2)
        val_filter = (x_df["ds"] >= d2) & (x_df["ds"] < d3)

        # # retrieve hyperparams
        # cpscale = x_df["changepoint_prior_scale"].iloc[0]
        # spscale = x_df["seasonality_prior_scale"].iloc[0]
        # hpscale = x_df["holidays_prior_scale"].iloc[0]
        # cr = x_df["changepoint_range"].iloc[0]

        # aggregate x_df on ds and key cols (already has all same key_cols so this is implicit)
        x_df = x_df.groupby("ds").agg({"y":"sum", "onpromotion":"sum", "dcoilwtico":"first"}).reset_index().sort_values("ds")

        # fit prophet model on train data
        model = Prophet(
            uncertainty_samples=0,
            holidays=h_df,
            # changepoint_prior_scale=cpscale,
            # seasonality_prior_scale=spscale,
            # holidays_prior_scale=hpscale,
            # changepoint_range=cr
        )
        model.add_regressor("onpromotion")
        model.add_regressor("dcoilwtico")
        model.fit(x_df[train_filter])

        # predict over train dates
        x_fit_df = model.predict(x_df[train_filter][["ds", "onpromotion", "dcoilwtico"]])[["ds", "yhat"]]
        x_fit_df["ds"] = np.datetime_as_string(x_fit_df["ds"].to_numpy(), unit='D')

        return x_fit_df

    def fit_lm(x_df, d1, d2, d3):
        x_df = x_df.sort_values("ds")
        train_filter = (x_df["ds"] >= d1) & (x_df["ds"] < d2)
        val_filter = (x_df["ds"] >= d2) & (x_df["ds"] < d3)

        X_train = x_df[train_filter][yhat_names].to_numpy()
        y_train = x_df[train_filter]["y"].to_numpy()
        model = Ridge(alpha=1e-2)
        model.fit(X_train, y_train)

        X_val = x_df[val_filter][yhat_names].to_numpy()
        y_val = x_df[val_filter]["y"].to_numpy()
        yhat = model.predict(X_val)
        ds = pd.date_range(start=d2, periods=val_days, freq="D", inclusive="left").strftime("%Y-%m-%d")
        df_dict = {"ds":ds, "yhat":yhat, "y":y_val}
        return pd.DataFrame(df_dict)

    # cv loop
    d = d0
    errs = []
    while d + timedelta(days=train_days) + timedelta(days=val_days - 1) <= d_max:
        # calculate train dates and test dates
        d1 = d.strftime("%Y-%m-%d")
        d2 = (d + timedelta(days=train_days)).strftime("%Y-%m-%d")
        d3 = (d + timedelta(days=train_days + val_days)).strftime("%Y-%m-%d")

        # get train and test sets
        # iterate over key_cols, hp_df pairs
        fits_dfs = []
        yhat_names = []
        for i in len(key_cols_list):
            key_cols = key_cols_list[i]
            hp_df = hp_df_list[i]
            # merge with correct hp_df
            fold_train_df = train_df.merge(hp_df, how="left", on=key_cols)
            # fit prophet model using hyperparams and get fitted values (using apply)
            fold_fits_df = fold_train_df.group_by(key_cols).apply(lambda x: fit_prophet(x, d1, d2, d3)).reset_index()
            yhat_name = "yhat" + "_".join(key_cols)
            fold_fits_df = fold_fits_df.rename({"yhat":yhat_name}, axis=1)
            # store fitted values in list
            fits_dfs.append(fold_fits_df)
            yhat_names.append(yhat_name)

        # merge all fit_dfs in the list
        fits_df = reduce(lambda x, y: x.merge(y, how="left", on=all_key_cols + ["ds"]), fits_dfs)
        fits_df = fits_df.merge(train_df, on=all_key_cols + ["ds"])

        # fit linear model to aggregate forecasts and predict (using apply)
        preds_df = fits_df.groupby(all_key_cols).apply(lambda x: fit_lm(x, d1, d2, d3)).reset_index()

        # calculate rmsle
        err = msle(preds_df)**0.5
        errs.append(err)

        d += timedelta(days=interval)
    
    return np.mean(np.array(errs))