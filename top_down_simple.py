from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from prophet import Prophet
from prophet.diagnostics import cross_validation

import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
# logging.getLogger("pandas").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True

def msle(preds_df):
    return np.mean((np.log(1 + preds_df["y"].values) - np.log(1 + preds_df["yhat"].values))**2)


def get_proportions(key_cols, support_cols, x_df):
    x_df = x_df.groupby(key_cols + support_cols + ["ds"]).agg({"y":"sum", "onpromotion":"sum", "dcoilwtico":"first"}).reset_index()
    agg_x_df = x_df.groupby(key_cols + ["ds"]).agg({"y":"sum"}).reset_index().rename({"y":"agg_y"}, axis=1)
    x_df = x_df.merge(agg_x_df, how="left", on=key_cols + ["ds"])
    x_df["prop"] = x_df.loc[:, ["y"]].where(x_df["agg_y"] <= 0, x_df["y"] / x_df["agg_y"], axis=0)
    return x_df.drop(["y", "agg_y"], axis=1)

def cross_validation(key_cols, support_cols, hp_df, train_df, stores_dict, all_holidays_df, max_window_size=14, train_days=365*3, val_days=16, interval=64):
    # aggregate train_df to lowest level to get truths
    truth_df = train_df.groupby(key_cols + support_cols + ["ds"]).agg({"y":"sum"}).reset_index().sort_values(key_cols + support_cols + ["ds"])
    # generate prop_df
    train_prop_df = get_proportions(key_cols, support_cols, train_df)

    def fit_predict_prophet(x_df, d1, d2, d3):
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

        t_df = x_df[train_filter]
        v_df = x_df[val_filter]

        # remove up to first nonzero entry of train_data
        first_date = t_df["ds"][t_df["y"] > 0].min()
        t_df = t_df[t_df["ds"] >= first_date]

        if t_df.shape[0] == 0:
            # when there is no data, return all zero predictions
            x_pred_df = v_df[["ds"]]
            x_pred_df["yhat"] = 0
        else:
            # otherwise, train model and predict over val dates
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
            model.fit(t_df)

            # predict over train and val dates
            x_pred_df = model.predict(v_df[["ds", "onpromotion", "dcoilwtico"]])[["ds", "yhat"]]
            # clip yhat
            x_pred_df["yhat"] = x_pred_df["yhat"].clip(0)
            x_pred_df["ds"] = np.datetime_as_string(x_pred_df["ds"].to_numpy(), unit='D')
        # add truth values
        x_pred_df = x_pred_df.merge(v_df[["ds", "y"]], how="inner", on=["ds"])

        return x_pred_df

    def roll_forward(x_df, d2):
        # x_df is prop_df grouped at lowest level

        # get max_window_size most recent train data points
        window_start_date = (datetime.strptime(d2, "%Y-%m-%d").date() - timedelta(days=max_window_size)).strftime("%Y-%m-%d")
        window = x_df[(x_df["ds"] >= window_start_date) & (x_df["ds"] < d2)]
        window_length = window.shape[0]
        pred_df = pd.DataFrame({
            "ds":pd.date_range(start=window_start_date, periods=val_days + window_length, freq="D", inclusive="left").strftime("%Y-%m-%d"),
            "prop_hat":np.concatenate([window["prop"].to_numpy(), np.zeros((val_days, ))])
        })
        if window_length > 0:
            window_sum = window["prop"].sum()
            for i in range(val_days):
                pred_df.iloc[i + window_length, 1] = window_sum / window_length
                window_sum -= pred_df.iloc[i, 1]
                window_sum += pred_df.iloc[i + window_length, 1]
        pred_df = pred_df.iloc[window_length:, :]
        return pred_df

    # merge train_df with hp_df
    train_df = train_df.merge(hp_df, how="left", on=key_cols)
    # we will wait for the model training apply call to aggregate train data and generate prop data, respectively

    # initialize earliest date
    d0 = datetime.strptime(train_df["ds"].iloc[0], "%Y-%m-%d").date()
    d_max = datetime.strptime(train_df["ds"].iloc[-1], "%Y-%m-%d").date()

    # cv loop
    d = d0
    msles_df_list = []
    f = 1
    while d + timedelta(days=train_days) + timedelta(days=val_days - 1) <= d_max:
        print("FOLD {}".format(f))
        f += 1
        # calculate boundary dates
        d1 = d.strftime("%Y-%m-%d")
        d2 = (d + timedelta(days=train_days)).strftime("%Y-%m-%d")
        d3 = (d + timedelta(days=train_days + val_days)).strftime("%Y-%m-%d")

        # fit prophet model at key_col level and generate predictions
        key_level_pred_df = train_df.groupby(key_cols)[train_df.columns].apply(lambda x: fit_predict_prophet(x, d1, d2, d3)).reset_index()
        # compute error at key level
        key_level_err = msle(key_level_pred_df)**0.5
        print("{} level error: {}".format(key_cols, key_level_err))
        # drop useless column now
        key_level_pred_df = key_level_pred_df.drop("y", axis=1)

        # roll average of last 14 days or so to calculate new props
        prop_preds_df = train_prop_df.groupby(key_cols + support_cols).apply(lambda x: roll_forward(x, d2), include_groups=False).reset_index()

        # merge prophet and var predictions at key_col level and multiply to calculate lowest level predictions
        pred_df = prop_preds_df.merge(key_level_pred_df, on=key_cols + ["ds"], how="left")
        pred_df["yhat"] = (pred_df["yhat"] * pred_df["prop_hat"])
        pred_df = pred_df.drop("prop_hat", axis=1)
        pred_df["yhat"] = pred_df["yhat"].clip(lower=0)

        # merge predictions with truth and calculate rmsle
        pred_df = pred_df.merge(truth_df, how="inner", on=key_cols + support_cols + ["ds"])
        fold_msles_df = pred_df.groupby(key_cols + support_cols).apply(msle, include_groups=False).reset_index().rename({0:"msle"}, axis=1)
        fold_msles_df["cutoff"] = d2
        msles_df_list.append(fold_msles_df)
        err = msle(pred_df)**0.5
        print("{} level error: {}".format(key_cols + support_cols, err))

        d += timedelta(days=interval)

    msles_df = pd.concat(msles_df_list)
    msles_df = msles_df.pivot(index=key_cols + support_cols, columns="cutoff", values="msle")
    return msles_df

def fit_predict(key_cols, train_df, test_df, stores_dict, all_holidays_df, hp_df, max_window_size=14):

    # generate prop_df
    train_prop_df = get_proportions(key_cols, support_cols, train_df)

    # merge train_df with hp_df
    train_df = train_df.merge(hp_df, on=key_cols)

    # group by key col and apply custom function
    def fp_prophet(x_df):
        # get key_col values for display purposes
        # create filter for test data
        tup = tuple()
        filter = pd.Series(test_df.shape[0] * [True])
        for c in key_cols:
            tup += (x_df[c].iloc[0], )
            filter = filter & (test_df[c] == x_df[c].iloc[0])
        
        # get test_data (dates and external regressors)
        x_test_df = test_df[filter][["ds", "onpromotion", "dcoilwtico"]].reset_index(drop=True)
        # agg x_test_df
        x_test_df = x_test_df.groupby("ds").agg({"y":"sum", "onpromotion":"sum", "dcoilwtico":"first"}).reset_index()

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

        # get hyperparams
        cp_scale = x_df["changepoint_prior_scale"].iloc[0]
        sp_scale = x_df["seasonality_prior_scale"].iloc[0]
        hp_scale = x_df["holidays_prior_scale"].iloc[0]
        cr = x_df["changepoint_range"].iloc[0]

        # aggregate x_df
        x_df = x_df.groupby("ds").agg({"y":"sum", "onpromotion":"sum", "dcoilwtico":"first"}).reset_index()

        # remove up to first nonzero entry of x_df
        first_date = x_df["ds"][x_df["y"] > 0].min()
        x_df = x_df[x_df["ds"] >= first_date]

        if x_df.shape[0] == 0:
            # when there is no data, return all zero predictions
            preds = x_test_df[["ds"]]
            preds["yhat"] = 0
        else:
            # instantiate model
            model = Prophet(
                uncertainty_samples=0,
                holidays=h_df,
                changepoint_prior_scale=cp_scale,
                seasonality_prior_scale=sp_scale,
                holidays_prior_scale=hp_scale,
                changepoint_range=cr
            )
            model.add_regressor("onpromotion")
            model.add_regressor("dcoilwtico")
            # fit model
            model.fit(x_df)
            preds = model.predict(x_test_df)[["ds", "yhat"]]
        # display string
        display_str = "{} predictions complete".format(tup)
        print(display_str)
        return preds
    
    def roll_forward(x_df):
        # x_df is prop_df grouped at lowest level

        # test df filter
        filter = pd.Series(test_df.shape[0] * [True])
        for c in key_cols + support_cols:
            filter = filter & (test_df[c] == x_df[c].iloc[0])

        # get test_data (dates and external regressors)
        x_test_df = test_df[filter][["ds", "onpromotion", "dcoilwtico"]].reset_index(drop=True)
        # agg x_test_df
        x_test_df = x_test_df.groupby("ds").agg({"y":"sum", "onpromotion":"sum", "dcoilwtico":"first"}).reset_index()

        latest_train_date = datetime.strptime(x_df["ds"].sort_values().iloc[-1], "%Y-%m-%d").date()
        latest_test_date = datetime.strptime(x_test_df["ds"].sort_values().iloc[-1], "%Y-%m-%d").date()
        roll_days = (latest_test_date - latest_train_date).days

        window_start_date = latest_train_date - timedelta(days=max_window_size).strftime("%Y-%m-%d")

        # get max_window_size most recent train data points
        window = x_df[x_df["ds"] >= window_start_date]
        window_length = window.shape[0]
        pred_df = pd.DataFrame({
            "ds":pd.date_range(start=window_start_date, periods=roll_days + window_length, freq="D", inclusive="left").strftime("%Y-%m-%d"),
            "prop_hat":np.concatenate([window["prop"].to_numpy(), np.zeros((roll_days, ))])
        })
        if window_length > 0:
            window_sum = window["prop"].sum()
            for i in range(roll_days):
                pred_df.iloc[i + window_length, 1] = window_sum / window_length
                window_sum -= pred_df.iloc[i, 1]
                window_sum += pred_df.iloc[i + window_length, 1]
        pred_df = pred_df.iloc[window_length:, :]
        return pred_df
    
    # fit predict prophet at key_col level
    key_level_pred_df = train_df.groupby(key_cols)[train_df.columns].apply(fp_prophet).reset_index()

    # roll props forward
    prop_preds_df = train_prop_df.groupby(key_cols + support_cols)[train_prop_df.columns].apply(roll_forward).reset_index()

    # merge prophet and var predictions at key_col level and multiply to calculate lowest level predictions
    pred_df = prop_preds_df.merge(key_level_pred_df, on=key_cols + ["ds"])
    pred_df["yhat"] = (pred_df["yhat"] * pred_df["prop_hat"])
    pred_df["yhat"] = pred_df["yhat"].clip(lower=0)
    pred_df = pred_df.reset_index()[key_cols + support_cols + ["ds", "yhat"]]
    return pred_df

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

    # cross validation using found hyperparameters
    key_cols = ["cluster", "family"]
    support_cols = ["store_nbr"]
    hp_df = pd.read_csv("./hyperparams/cluster_family_hyperparams2.csv")
    msles_df = cross_validation(key_cols, support_cols, hp_df, train_df, stores_dict, all_holidays_df, max_window_size=14, train_days=365*4 + 128, val_days=16, interval=28)
    msles_df.to_csv("./top_down_store_nbr_family_best_msles2.csv")
    print(msles_df.mean(axis=0)**0.5)