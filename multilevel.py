from datetime import date, datetime, timedelta
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
# logging.getLogger("pandas").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True

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

        # retrieve hyperparams
        cpscale = x_df["changepoint_prior_scale"].iloc[0]
        spscale = x_df["seasonality_prior_scale"].iloc[0]
        hpscale = x_df["holidays_prior_scale"].iloc[0]
        cr = x_df["changepoint_range"].iloc[0]

        # aggregate x_df on ds and key cols (already has all same key_cols so this is implicit)
        x_df = x_df.groupby("ds").agg({"y":"sum", "onpromotion":"sum", "dcoilwtico":"first"}).reset_index().sort_values("ds")

        train_filter = (x_df["ds"] >= d1) & (x_df["ds"] < d2)
        val_filter = (x_df["ds"] >= d2) & (x_df["ds"] < d3)

        # fit prophet model on train data
        model = Prophet(
            uncertainty_samples=0,
            holidays=h_df,
            changepoint_prior_scale=cpscale,
            seasonality_prior_scale=spscale,
            holidays_prior_scale=hpscale,
            changepoint_range=cr
        )
        model.add_regressor("onpromotion")
        model.add_regressor("dcoilwtico")
        model.fit(x_df[train_filter])

        # predict over train and val dates
        x_fit_df = model.predict(x_df[["ds", "onpromotion", "dcoilwtico"]])[["ds", "yhat"]]
        # consider clipping yhat
        x_fit_df["yhat"] = x_fit_df["yhat"].clip(0)
        x_fit_df["ds"] = np.datetime_as_string(x_fit_df["ds"].to_numpy(), unit='D')

        return x_fit_df

    def fit_lm(x_df, d1, d2, d3, yhat_names):
        x_df = x_df.sort_values("ds")
        train_filter = (x_df["ds"] >= d1) & (x_df["ds"] < d2)
        val_filter = (x_df["ds"] >= d2) & (x_df["ds"] < d3)

        X_train = x_df[train_filter][yhat_names].to_numpy()
        y_train = (x_df[train_filter]["y"].to_numpy() - x_df[train_filter]["yhat_store_nbr_family"].to_numpy())[:, np.newaxis]

        scaler = StandardScaler()
        scaler.fit(X_train)

        model = Ridge(alpha=1e1)
        model.fit(scaler.transform(X_train), y_train)

        X_val = x_df[val_filter][yhat_names].to_numpy()
        y_val = x_df[val_filter]["y"].to_numpy()
        yhat = model.predict(scaler.transform(X_val)).flatten()
        yhat = yhat + x_df[val_filter]["yhat_store_nbr_family"].to_numpy()
        yhat[yhat < 0] = 0
        ds = pd.date_range(start=d2, periods=val_days, freq="D", inclusive="left").strftime("%Y-%m-%d")
        df_dict = {"ds":ds, "yhat":yhat, "y":y_val}
        return pd.DataFrame(df_dict)

    # cv loop
    d = d0
    errs = []
    f = 1
    while d + timedelta(days=train_days) + timedelta(days=val_days - 1) <= d_max:
        print("FOLD {}".format(f))
        f += 1
        # calculate train dates and test dates
        d1 = d.strftime("%Y-%m-%d")
        d2 = (d + timedelta(days=train_days)).strftime("%Y-%m-%d")
        d3 = (d + timedelta(days=train_days + val_days)).strftime("%Y-%m-%d")

        # iterate over key_cols, hp_df pairs
        fits_dfs = []
        yhat_names = []
        for i in range(len(key_cols_list)):
            key_cols = key_cols_list[i]
            hp_df = hp_df_list[i]
            # merge with correct hp_df
            fold_train_df = train_df.merge(hp_df, how="left", on=key_cols)
            # fit prophet model using hyperparams and get fitted values (using apply)
            fold_fits_df = fold_train_df.groupby(key_cols)[fold_train_df.columns].apply(lambda x: fit_prophet(x, d1, d2, d3)).reset_index()
            yhat_name = "yhat_" + "_".join(key_cols)
            fold_fits_df = fold_fits_df.rename({"yhat":yhat_name}, axis=1).drop("level_{}".format(len(key_cols)), axis=1)
            # store fitted values in list
            fits_dfs.append(fold_fits_df)
            yhat_names.append(yhat_name)
            # calculate baseline_err
            y_val = train_df[(train_df["ds"] >= d2) & (train_df["ds"] < d3)].groupby(key_cols + ["ds"]).agg({"y":"sum"}).reset_index().sort_values(key_cols + ["ds"])["y"]
            yhat_val = fold_fits_df[(fold_fits_df["ds"] >= d2) & (fold_fits_df["ds"] < d3)].reset_index(drop=True).sort_values(key_cols + ["ds"])[yhat_name]
            baseline_df = pd.DataFrame({"y":y_val, "yhat":yhat_val})
            baseline_err = msle(baseline_df)**0.5
            print("{} baseline rmsle: {}".format(key_cols, baseline_err))

        # merge all fit_dfs in the list
        fits_df = train_df[all_key_cols + ["ds", "y"]]
        for i, fold_fits_df in enumerate(fits_dfs):
            key_cols = key_cols_list[i]
            fits_df = fits_df.merge(fold_fits_df, on=key_cols + ["ds"])
        

        # fit linear model to aggregate forecasts and predict (using apply)
        preds_df = fits_df.groupby(all_key_cols)[fits_df.columns].apply(lambda x: fit_lm(x, d1, d2, d3, yhat_names)).reset_index()

        # calculate rmsle
        err = msle(preds_df)**0.5
        print(err)
        errs.append(err)

        d += timedelta(days=interval)
    
    return np.mean(np.array(errs))

if __name__ == "__main__":
    # data loading and processing
    train_df = pd.read_csv("./train.csv")
    train_df = train_df.rename({"date":"ds", "sales":"y"}, axis=1)

    oil_df = pd.read_csv("oil.csv")
    oil_df = oil_df.rename({"date":"ds"}, axis=1)

    stores_df = pd.read_csv("stores.csv")
    stores_dict = stores_df.set_index("store_nbr").to_dict("index")

    holidays_df = pd.read_csv("holidays_events.csv")

    test_df = pd.read_csv("test.csv")
    test_df = test_df.rename({"date":"ds"}, axis=1)

    # # subset train data for debugging
    # subset_families = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY']
    # train_df = train_df[(train_df["store_nbr"] < 4) & (train_df["family"].isin(subset_families))]

    ## interpolate missing oil values
    blank_oil_df = pd.DataFrame({"ds":pd.date_range(train_df["ds"].min(), test_df["ds"].max()).astype("str")})
    oil_df = blank_oil_df.merge(oil_df, how="left", on="ds")
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].interpolate("nearest")
    oil_df.iloc[0, 1] = 93.14
    train_df = train_df.merge(oil_df, how="left", on="ds")
    test_df = test_df.merge(oil_df, how="left", on="ds")

    ## processing holidays df
    nth_df = holidays_df[holidays_df["transferred"] == False]
    th_df = holidays_df[holidays_df["type"] == "Transfer"]
    th_df["description"] = th_df["description"].str.removeprefix("Traslado ")
    all_holidays_df = pd.concat([nth_df, th_df], axis=0)[["date", "locale_name", "description"]]
    all_holidays_df = all_holidays_df.rename({"date":"ds", "description":"holiday"}, axis=1)
    all_holidays_df["lower_window"] = 0
    all_holidays_df["upper_window"] = 1

    ## info cols
    train_df = train_df.merge(stores_df[["store_nbr", "cluster"]], how="left", on="store_nbr")
    test_df = test_df.merge(stores_df[["store_nbr", "cluster"]], how="left", on="store_nbr")

    # setup
    key_cols_list = [
        ["cluster"],
        ["store_nbr"],
        ["family"],
        ["cluster", "family"],
        ["store_nbr", "family"]
    ]

    c_hp_df = pd.read_csv("./cluster_hyperparams.csv")
    s_hp_df = pd.read_csv("./store_nbr_hyperparams.csv")
    f_hp_df = pd.read_csv("./family_hyperparams.csv")
    cf_hp_df = pd.read_csv("./cluster_family_hyperparams.csv")

    hp_df1 = pd.read_csv("./store_nbr_family_hyperparams_1.csv")
    hp_df2 = pd.read_csv("./store_nbr_family_hyperparams_2.csv")
    hp_df3 = pd.read_csv("./store_nbr_family_hyperparams_3.csv")
    hp_df4 = pd.read_csv("./store_nbr_family_hyperparams_4.csv")
    hp_df5 = pd.read_csv("./store_nbr_family_hyperparams_5.csv")
    sf_hp_df = pd.concat([hp_df1, hp_df2, hp_df3, hp_df4, hp_df5], axis=0)

    hp_df_list = [
        c_hp_df,
        s_hp_df,
        f_hp_df,
        cf_hp_df,
        sf_hp_df
    ]

    cv_err = cross_validation(train_df, key_cols_list, hp_df_list, stores_dict, all_holidays_df, train_days=365*4, val_days=16, interval=64)
    print(cv_err)