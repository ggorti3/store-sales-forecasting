from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from prophet import Prophet
from prophet.diagnostics import cross_validation as prophet_cross_validation

import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
# logging.getLogger("pandas").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True

def msle(preds_df):
    return np.mean((np.log(1 + preds_df["y"].values) - np.log(1 + preds_df["yhat"].values))**2)

def visualize(key_col_values, train_df, hp_df, cutoffs):
    key_cols = list(key_col_values.keys())

    # filter correct key_col values
    hp_filter = pd.Series(np.full((hp_df.shape[0], ), True, dtype=np.bool_))
    data_filter = pd.Series(np.full((train_df.shape[0], ), True, dtype=np.bool_))
    for c, v in key_col_values.items():
        data_filter = data_filter & (train_df[c] == v)
        hp_filter = hp_filter & (hp_df[c] == v)
    hps = hp_df[hp_filter]
    train_df = train_df[data_filter]

    # get hyperparameters
    cp_scale = hps["changepoint_prior_scale"].iloc[0]
    hp_scale = hps["holidays_prior_scale"].iloc[0]
    sp_scale = hps["seasonality_prior_scale"].iloc[0]
    cr = hps["changepoint_range"].iloc[0]

    # get h_df
    store_nbrs = train_df["store_nbr"].drop_duplicates()
    states = [stores_dict[snbr]["state"] for snbr in store_nbrs]
    cities = [stores_dict[snbr]["city"] for snbr in store_nbrs]
    filter = (all_holidays_df["locale_name"] == "Ecuador")
    for s in states:
        filter = filter | (all_holidays_df["locale_name"] == s)
    for c in cities:
        filter = filter | (all_holidays_df["locale_name"] == c)
    h_df = all_holidays_df[filter]
    h_df = h_df[["ds", "holiday", "lower_window", "upper_window"]]
    # aggregate train_df
    train_df = train_df.groupby(key_cols + ["ds"]).agg({"y":"sum", "onpromotion":"sum", "dcoilwtico":"first"}).reset_index()
    data = train_df.sort_values("ds")[["ds", "y", "dcoilwtico", "onpromotion"]]

    preds_list = []

    for cutoff in cutoffs:
        # get train and val data
        train_data = data[data["ds"] <= cutoff]

        # fit prophet model
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
        model.fit(train_data)
        # forecast over train and val dates
        preds = model.predict(data.drop("y", axis=1))[["ds", "yhat"]]
        preds_list.append(preds)

    # plot truth and predictions
    fig, axes = plt.subplots(len(cutoffs), 1)
    for i, preds in enumerate(preds_list):
        ax = axes[i]
        ax.plot(pd.to_datetime(data["ds"]), data["y"])
        ax.plot(pd.to_datetime(preds["ds"]), preds["yhat"])
    axes[0].legend(["truth", "predicted"])
    plt.show()

    pass

def cv_optimize(key_cols, train_df, stores_dict, all_holidays_df, hparam_grid, train_days=365*4 + 128, val_days=16, interval=28):
    d0 = datetime.strptime(train_df["ds"].iloc[0], "%Y-%m-%d").date()
    d_max = datetime.strptime(train_df["ds"].iloc[-1], "%Y-%m-%d").date()

    def cv_opt(x_df):
        # get key_col values for display purposes
        tup = tuple()
        for c in key_cols:
            tup += (x_df[c].iloc[0], )

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
                # manual cv loop
                d = d0
                preds_df_list = []
                while d + timedelta(days=train_days) + timedelta(days=val_days - 1) <= d_max:
                    d1 = d.strftime("%Y-%m-%d")
                    d2 = (d + timedelta(days=train_days)).strftime("%Y-%m-%d")
                    d3 = (d + timedelta(days=train_days + val_days)).strftime("%Y-%m-%d")

                    fold_train_df = x_df[(x_df["ds"] >= d1) & (x_df["ds"] < d2)]
                    fold_val_df = x_df[(x_df["ds"] >= d2) & (x_df["ds"] < d3)]

                    # remove up to first nonzero entry pf train_data
                    first_date = fold_train_df["ds"][fold_train_df["y"] > 0].min()
                    fold_train_df = fold_train_df[fold_train_df["ds"] >= first_date]


                    if fold_train_df.shape[0] == 0:
                        # when there is no data, return all zero predictions
                        fold_preds_df = fold_val_df[["ds"]]
                        fold_preds_df["yhat"] = 0
                        fold_preds_df["y"] = fold_val_df["y"]
                    else:
                        # otherwise, train model and predict over val dates
                        model = Prophet(
                            uncertainty_samples=0,
                            holidays=h_df,
                            **hparams
                        )
                        model.add_regressor("onpromotion")
                        model.add_regressor("dcoilwtico")
                        model.fit(fold_train_df)

                        fold_preds_df = model.predict(fold_val_df[["ds", "onpromotion", "dcoilwtico"]])[["ds", "yhat"]]
                        fold_preds_df["ds"] = np.datetime_as_string(fold_preds_df["ds"].to_numpy(), unit='D')
                        fold_preds_df = fold_preds_df.merge(fold_val_df, on=["ds"]).drop(["onpromotion", "dcoilwtico"], axis=1)
                    fold_preds_df["cutoff"] = d2
                    preds_df_list.append(fold_preds_df)
                    d += timedelta(days=interval)
                cv_df = pd.concat(preds_df_list, axis=0)
                cv_df["yhat"] = cv_df["yhat"].clip(lower=0)
                msles = cv_df.groupby("cutoff")[cv_df.columns].apply(msle).reset_index()
                err = msles.iloc[:, 1].mean()**0.5
                if err < min_err:
                    best_hparams[hp] = val
                    min_err = err
        
        print("{} min cv rmsle: {}".format(tup, min_err))
        for hp in best_hparams.keys():
            print("    {}: {}".format(hp, best_hparams[hp]))
        df_dict = {hp:[best_hparams[hp]] for hp in best_hparams.keys()}
        df_dict["cv_rmsle"] = [min_err]
        return pd.DataFrame(df_dict)

    best_hparams_df = train_df.groupby(key_cols).apply(cv_opt).reset_index()
    return best_hparams_df

def cross_validation(key_cols, train_df, stores_dict, all_holidays_df, hp_df, train_days=365*4 + 128, val_days=16, interval=28):
    # merge train_df with hp_df
    train_df = train_df.merge(hp_df, on=key_cols)

    d0 = datetime.strptime(train_df["ds"].iloc[0], "%Y-%m-%d").date()
    d_max = datetime.strptime(train_df["ds"].iloc[-1], "%Y-%m-%d").date()
    

    def cv(x_df):
        # get key_col values for display purposes
        tup = tuple()
        for c in key_cols:
            tup += (x_df[c].iloc[0], )

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

        # manual cv loop
        d = d0
        preds_df_list = []
        while d + timedelta(days=train_days) + timedelta(days=val_days - 1) <= d_max:
            d1 = d.strftime("%Y-%m-%d")
            d2 = (d + timedelta(days=train_days)).strftime("%Y-%m-%d")
            d3 = (d + timedelta(days=train_days + val_days)).strftime("%Y-%m-%d")

            fold_train_df = x_df[(x_df["ds"] >= d1) & (x_df["ds"] < d2)]
            fold_val_df = x_df[(x_df["ds"] >= d2) & (x_df["ds"] < d3)]

            # remove up to first nonzero entry pf train_data
            first_date = fold_train_df["ds"][fold_train_df["y"] > 0].min()
            fold_train_df = fold_train_df[fold_train_df["ds"] >= first_date]


            if fold_train_df.shape[0] == 0:
                # when there is no data, return all zero predictions
                fold_preds_df = fold_val_df[["ds"]]
                fold_preds_df["yhat"] = 0
                fold_preds_df["y"] = fold_val_df["y"]
            else:
                # otherwise, train model and predict over val dates
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
                model.fit(fold_train_df)

                fold_preds_df = model.predict(fold_val_df[["ds", "onpromotion", "dcoilwtico"]])[["ds", "yhat"]]
                fold_preds_df["ds"] = np.datetime_as_string(fold_preds_df["ds"].to_numpy(), unit='D')
                fold_preds_df = fold_preds_df.merge(fold_val_df, on=["ds"]).drop(["onpromotion", "dcoilwtico"], axis=1)
            fold_preds_df["cutoff"] = d2
            preds_df_list.append(fold_preds_df)
            d += timedelta(days=interval)
        
        cv_df = pd.concat(preds_df_list, axis=0)
        cv_df["yhat"] = cv_df["yhat"].clip(lower=0)
        msles = cv_df.groupby("cutoff")[cv_df.columns].apply(msle).reset_index()
        # display string
        display_str = "{}: ".format(tup)
        for i in range(msles.shape[0]):
            display_str += "{:0.5f} ".format(msles.iloc[i, 1]**0.5)
        print(display_str)
        return msles

    msles_df = train_df.groupby(key_cols)[train_df.columns].apply(cv).reset_index()
    msles_df = msles_df.drop("level_{}".format(len(key_cols)), axis=1).rename({0:"msle"}, axis=1)
    msles_df = msles_df.pivot(columns="cutoff", index=key_cols, values="msle")
    return msles_df

def fit_predict(key_cols, train_df, test_df, stores_dict, all_holidays_df, hp_df):
    # merge train_df with hp_df
    train_df = train_df.merge(hp_df, on=key_cols)

    # group by key col and apply custom function
    def fp(x_df):
        # get key_col values for display purposes
        # create filter for test data
        tup = tuple()
        filter = pd.Series(test_df.shape[0] * [True])
        for c in key_cols:
            tup += (x_df[c].iloc[0], )
            filter = filter & (test_df[c] == x_df[c].iloc[0])
        
        # get test_data (dates and external regressors)
        x_test_df = test_df[filter][["ds", "onpromotion", "dcoilwtico"]].reset_index(drop=True)

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

    return train_df.groupby(key_cols)[train_df.columns].apply(fp).reset_index()

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
    # train_df = train_df[(train_df["store_nbr"] > 45)]

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

    # # set variables
    # key_cols = ["family"]
    # hparam_grid = {
    #     "changepoint_prior_scale":[0.001, 0.01, 0.1, 0.5],
    #     "seasonality_prior_scale":[0.01, 0.1, 1, 10],
    #     "holidays_prior_scale":[0.01, 0.1, 1, 10],
    #     "changepoint_range":[0.8, 0.9, 0.95]
    # }

    # # run hyperparameter optimization
    # hparams_df = cv_optimize(key_cols, train_df, stores_dict, all_holidays_df, hparam_grid, train_days=365*4 + 128, val_days=16, interval=28)
    # hparams_df.to_csv("./family_hyperparams2.csv")

    # # cross validation using found hyperparameters
    # key_cols = ["store_nbr", "family"]
    # hp_df = pd.read_csv("./store_nbr_family_hyperparams2.csv")
    # msles_df = cross_validation(key_cols, train_df, stores_dict, all_holidays_df, hp_df, train_days=365*4 + 128, val_days=16, interval=28)
    # msles_df.to_csv("./store_nbr_family_best_msles2.csv")
    # print(msles_df.mean(axis=0)**0.5)

    # rand_store_nbr = 24
    # rand_family = ""
    # key_col_values = {
    #     "store_nbr":12,
    #     "family":"BEVERAGES"
    # }
    # hp_df = pd.read_csv("./hyperparams/store_nbr_family_hyperparams.csv")
    # cutoffs = ["2017-02-12", "2017-04-09", "2017-06-04", "2017-07-30"]
    # visualize(key_col_values, train_df, hp_df, cutoffs)

    # fit predict
    key_cols = ["store_nbr", "family"]
    hp_df = pd.read_csv("./hyperparams/store_nbr_family_hyperparams2.csv")
    pred_df = fit_predict(key_cols, train_df, test_df, stores_dict, all_holidays_df, hp_df)
    pred_df.to_csv("./single_level_predictions.csv")
    print(pred_df.head())