from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from prophet import Prophet
from prophet.diagnostics import cross_validation

def msle(preds_df):
    return np.mean((np.log(1 + preds_df["y"].values) - np.log(1 + preds_df["yhat"].values))**2)


def cv_optimize(key_cols, train_df, stores_dict, all_holidays_df, hparam_grid):
    def cv_opt(x_df):
        # get h_df
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

        # aggregate x_df
        x_df = x_df.groupby("ds").agg({"y":"sum", "onpromotion":"sum", "dcoilwtico":"first"}).reset_index()

        # hparam dict
        best_hparams = {hp:hparam_grid[hp][0] for hp in hparam_grid.keys()}
        # coord descent loop over hparam grid
        min_err = float("inf")
        for i, hp in enumerate(hparam_grid.keys()):
            # prevent redundant cvs
            grid = hparam_grid[hp] if i == 0 else hparam_grid[hp][1:]
            for val in grid:
                hparams = best_hparams.copy()
                hparams[hp] = val
                model = Prophet(uncertainty_samples=0, holidays=h_df, **hparams)
                model.add_regressor("onpromotion")
                model.add_regressor("dcoilwtico")
                model.fit(x_df)
                cv_df = cross_validation(model, initial='1460 days', period='56 days', horizon='16 days')
                cv_df["yhat"] = cv_df["yhat"].clip(lower=0)
                msles = cv_df.groupby("cutoff").apply(msle).values
                err = np.mean(msles)
                if err < min_err:
                    best_hparams[hp] = val
                    min_err = err
        df_dict = {hp:[best_hparams[hp]] for hp in best_hparams.keys()}
        df_dict["msle"] = [min_err]
        return pd.DataFrame(df_dict)

    best_hparams_df = train_df.groupby(key_cols).apply(cv_opt).reset_index()
    return best_hparams_df

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

    # subset train data for computational reasons
    # we will optimize hyperparams over smaller subsets of the data
    # there are 54 stores, we will do 10 at a time
    # train_df = train_df[(train_df["store_nbr"] >= 41)]

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

    # set variables
    # key_cols = ["store_nbr", "family"]
    # hparam_grid = {
    #     "changepoint_prior_scale":[0.001, 0.01, 0.1, 0.5],
    #     "seasonality_prior_scale":[0.01, 0.5, 10],
    #     "holidays_prior_scale":[0.01, 0.5, 10],
    #     "changepoint_range":[0.8, 0.9, 0.95]
    # }

    key_cols = ["cluster", "family"]
    hparam_grid = {
        "changepoint_prior_scale":[0.001, 0.01, 0.1, 0.5],
        "seasonality_prior_scale":[0.01, 0.5, 10],
        "holidays_prior_scale":[0.01, 0.5, 10],
        "changepoint_range":[0.8, 0.9, 0.95]
    }

    # run hyperparameter optimization
    hparams_df = cv_optimize(key_cols, train_df, stores_dict, all_holidays_df, hparam_grid)
    hparams_df.to_csv("./cluster_family_hyperparams.csv")