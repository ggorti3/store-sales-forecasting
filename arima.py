from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ValueWarning)
warnings.simplefilter('ignore', UserWarning)


def msle(preds_df):
    return np.mean((np.log(1 + preds_df["y"].values) - np.log(1 + preds_df["yhat"].values))**2)

def year_diff(key_cols, x_df):
    other_df = x_df.copy()
    other_df["ds"] = (pd.to_datetime(other_df["ds"]) + DateOffset(years=1)).dt.strftime("%Y-%m-%d")
    other_df = other_df.groupby(key_cols + ["ds"]).agg("first").reset_index()
    other_df = other_df.rename({"y":"prev_y"}, axis=1)[key_cols + ["ds", "prev_y"]]
    leap_yrs_df = other_df[other_df["ds"] == "2016-02-28"]
    leap_yrs_df["ds"] = "2016-02-29"
    other_df = pd.concat([other_df, leap_yrs_df]).sort_values(key_cols + ["ds"]).reset_index(drop=True)
    dx_df = x_df.merge(other_df, how="inner", on=key_cols + ["ds"])
    dx_df["y"] = dx_df["y"] - dx_df["prev_y"]
    dx_df = dx_df[key_cols + ["ds", "y", "onpromotion", "dcoilwtico"]]
    return dx_df.sort_values(key_cols + ["ds"]), other_df.sort_values(key_cols + ["ds"])

def year_inv_diff(key_cols, yhat_df, other_df):
    yhat_df = yhat_df.merge(other_df, how="inner", on=key_cols + ["ds"])
    yhat_df["yhat"] = yhat_df["yhat"] + yhat_df["prev_y"]
    yhat_df = yhat_df[key_cols + ["ds", "yhat"]]
    return yhat_df

def cv_optimize(key_cols, train_df, stores_dict, all_holidays_df, hparam_grid, train_days=365*4 + 128, val_days=16, interval=28):
    d0 = datetime.strptime(train_df["ds"].iloc[0], "%Y-%m-%d").date()
    d_max = datetime.strptime(train_df["ds"].iloc[-1], "%Y-%m-%d").date()

    def cv_opt(x_df):
        # get key_col values for display purposes
        tup = tuple()
        for c in key_cols:
            tup += (x_df[c].iloc[0], )

        # # get h_df
        # store_nbrs = x_df["store_nbr"].drop_duplicates()
        # states = [stores_dict[snbr]["state"] for snbr in store_nbrs]
        # cities = [stores_dict[snbr]["city"] for snbr in store_nbrs]
        # filter = (all_holidays_df["locale_name"] == "Ecuador")
        # for s in states:
        #     filter = filter | (all_holidays_df["locale_name"] == s)
        # for c in cities:
        #     filter = filter | (all_holidays_df["locale_name"] == c)
        # h_df = all_holidays_df[filter]
        # h_df = h_df[["ds", "holiday"]].sort_values(["ds", "holiday"])
        # h_df["dummy"] = 1
        # h_df = h_df.drop_duplicates()
        # one_hot_h_df = h_df.pivot(index="ds", columns="holiday", values="dummy").fillna(0).reset_index()

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
                # get hparam dict
                hparams = best_hparams.copy()
                hparams[hp] = val
                model_hparams = {}
                if hparams["diff"] == "day" or hparams["diff"] == "both":
                    model_hparams["order"] = (hparams["order"][0], 1, hparams["order"][1])
                else:
                    model_hparams["order"] = (hparams["order"][0], 0, hparams["order"][1])
                # manual cv loop
                d = d0
                preds_df_list = []
                while d + timedelta(days=train_days) + timedelta(days=val_days - 1) <= d_max:
                    d1 = d.strftime("%Y-%m-%d")
                    d2 = (d + timedelta(days=train_days)).strftime("%Y-%m-%d")
                    d3 = (d + timedelta(days=train_days + val_days)).strftime("%Y-%m-%d")

                    fold_train_df = x_df[(x_df["ds"] >= d1) & (x_df["ds"] < d2)]
                    fold_val_df = x_df[(x_df["ds"] >= d2) & (x_df["ds"] < d3)]

                    if hparams["diff"] == "year" or hparams["diff"] == "both":
                        # difference train data
                        fold_train_df, fold_prev_df = year_diff([], fold_train_df)

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
                        
                        # convert ds to datetime
                        #fold_train_df["ds"] = pd.to_datetime(fold_train_df["ds"])
                        # instantiate clean daterange
                        fold_train_dates = pd.date_range(start=first_date, end=d2, inclusive="left")
                        fold_train_dates_df = pd.DataFrame({
                        "ds":fold_train_dates
                        })
                        fold_train_dates_df["ds"] = fold_train_dates_df["ds"].dt.strftime("%Y-%m-%d")
                        # merge train data with daterange
                        fold_train_df = fold_train_dates_df.merge(fold_train_df, on="ds", how="left")
                        fold_regressors_df = fold_train_df[["ds", "onpromotion", "dcoilwtico"]]
                        fold_train_df = fold_train_df[["ds", "y"]]
                        # interpolate missing y values (typically christmas?)
                        fold_train_df["y"] = fold_train_df["y"].interpolate("nearest")
                        # convert ds to datetime then set as index
                        fold_train_df["ds"] = pd.to_datetime(fold_train_df["ds"])
                        fold_train_df = fold_train_df.set_index("ds")

                        # interpolate missing regressor values
                        fold_regressors_df["onpromotion"] = fold_regressors_df["onpromotion"].interpolate("nearest")
                        fold_regressors_df["dcoilwtico"] = fold_regressors_df["dcoilwtico"].interpolate("nearest")
                        # merge regressors df with holidays df
                        #fold_regressors_df = fold_regressors_df.merge(one_hot_h_df, on="ds", how="left").fillna(0).sort_values("ds")

                        # # val regressors
                        # fold_val_dates = pd.date_range(start=d2, end=d3, inclusive="left")
                        # fold_val_dates_df = pd.DataFrame({
                        #    "ds":fold_val_dates
                        # })
                        # fold_val_dates_df["ds"] = fold_val_dates_df["ds"].dt.strftime("%Y-%m-%d")
                        # fold_val_regressors_df = fold_val_dates_df.merge(fold_val_df, on="ds", how="left")[["ds", "onpromotion", "dcoilwtico"]]
                        # fold_val_regressors_df["onpromotion"] = fold_val_regressors_df["onpromotion"].interpolate("nearest")
                        # fold_val_regressors_df["dcoilwtico"] = fold_val_regressors_df["dcoilwtico"].interpolate("nearest")
                        # fold_val_regressors_df = fold_val_regressors_df.merge(one_hot_h_df, on="ds", how="left").fillna(0).sort_values("ds")

                        model = SARIMAX(
                            endog=fold_train_df,
                            #exog=fold_regressors_df.iloc[:, 1:].values,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            **model_hparams
                        )
                        m_fit = model.fit(disp=0)
                        preds = m_fit.forecast(val_days)#, exog=fold_val_regressors_df.iloc[:, 1:].values)
                        preds = preds.to_frame().reset_index().rename({"index":"ds", "predicted_mean":"yhat"}, axis=1)
                        preds["ds"] = preds["ds"].dt.strftime("%Y-%m-%d")
                        if hparams["diff"] == "year" or hparams["diff"] == "both":
                            preds = year_inv_diff([], preds, fold_prev_df)
                        fold_preds_df = preds.merge(fold_val_df, on="ds")[["ds", "yhat", "y"]]

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

def cross_validation(key_cols, train_df, stores_dict, all_holidays_df, hp_df=None, train_days=365*4 + 128, val_days=16, interval=28):
    # merge train_df with hp_df
    if hp_df is not None:
        train_df = train_df.merge(hp_df, on=key_cols)

    d0 = datetime.strptime(train_df["ds"].iloc[0], "%Y-%m-%d").date()
    d_max = datetime.strptime(train_df["ds"].iloc[-1], "%Y-%m-%d").date()
    

    def cv(x_df):
        # get key_col values for display purposes
        tup = tuple()
        for c in key_cols:
            tup += (x_df[c].iloc[0], )

        # # get h_df
        # store_nbrs = x_df["store_nbr"].drop_duplicates()
        # states = [stores_dict[snbr]["state"] for snbr in store_nbrs]
        # cities = [stores_dict[snbr]["city"] for snbr in store_nbrs]
        # filter = (all_holidays_df["locale_name"] == "Ecuador")
        # for s in states:
        #     filter = filter | (all_holidays_df["locale_name"] == s)
        # for c in cities:
        #     filter = filter | (all_holidays_df["locale_name"] == c)
        # h_df = all_holidays_df[filter]
        # h_df = h_df[["ds", "holiday"]].sort_values(["ds", "holiday"])
        # h_df["dummy"] = 1
        # h_df = h_df.drop_duplicates()
        # one_hot_h_df = h_df.pivot(index="ds", columns="holiday", values="dummy").fillna(0).reset_index()

        hparams = {}
        if hp_df is not None:
            diff = x_df["diff"].iloc[0]
            order_str = x_df["order"].iloc[0]
            order = (int(order_str[1]), int(order_str[4]))
            if diff == "day" or diff == "both":
                order_tup = (order[0], 1, order[1])
            else:
                order_tup = (order[0], 0, order[1])
            hparams["order"] = order_tup

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
            fold_val_df = x_df[(x_df["ds"] >= d2) & (x_df["ds"] < d3)].sort_values("ds")

            if hp_df is not None and (diff == "year" or diff == "both"):
                # difference train data
                fold_train_df, fold_prev_df = year_diff([], fold_train_df)

            # remove up to first nonzero entry of train_data
            first_date = fold_train_df["ds"][fold_train_df["y"] > 0].min()
            fold_train_df = fold_train_df[fold_train_df["ds"] >= first_date]

            if fold_train_df.shape[0] == 0:
                # when there is no data, return all zero predictions
                fold_preds_df = fold_val_df[["ds"]]
                fold_preds_df["yhat"] = 0
                fold_preds_df["y"] = fold_val_df["y"]
            else:
                # otherwise, train model and predict over val dates
                
                # convert ds to datetime
                #fold_train_df["ds"] = pd.to_datetime(fold_train_df["ds"])
                # instantiate clean daterange
                fold_train_dates = pd.date_range(start=first_date, end=d2, inclusive="left")
                fold_train_dates_df = pd.DataFrame({
                   "ds":fold_train_dates
                })
                fold_train_dates_df["ds"] = fold_train_dates_df["ds"].dt.strftime("%Y-%m-%d")
                # merge train data with daterange
                fold_train_df = fold_train_dates_df.merge(fold_train_df, on="ds", how="left")
                fold_regressors_df = fold_train_df[["ds", "onpromotion", "dcoilwtico"]]
                fold_train_df = fold_train_df[["ds", "y"]]
                # interpolate missing y values (typically christmas?)
                fold_train_df["y"] = fold_train_df["y"].interpolate("nearest")
                # convert ds to datetime then set as index
                fold_train_df["ds"] = pd.to_datetime(fold_train_df["ds"])
                fold_train_df = fold_train_df.set_index("ds")

                # interpolate missing regressor values
                fold_regressors_df["onpromotion"] = fold_regressors_df["onpromotion"].interpolate("nearest")
                fold_regressors_df["dcoilwtico"] = fold_regressors_df["dcoilwtico"].interpolate("nearest")
                # # merge regressors df with holidays df
                # fold_regressors_df = fold_regressors_df.merge(one_hot_h_df, on="ds", how="left").fillna(0).sort_values("ds")

                # val regressors
                fold_val_dates = pd.date_range(start=d2, end=d3, inclusive="left")
                fold_val_dates_df = pd.DataFrame({
                   "ds":fold_val_dates
                })
                fold_val_dates_df["ds"] = fold_val_dates_df["ds"].dt.strftime("%Y-%m-%d")
                fold_val_regressors_df = fold_val_dates_df.merge(fold_val_df, on="ds", how="left")[["ds", "onpromotion", "dcoilwtico"]]
                fold_val_regressors_df["onpromotion"] = fold_val_regressors_df["onpromotion"].interpolate("nearest")
                fold_val_regressors_df["dcoilwtico"] = fold_val_regressors_df["dcoilwtico"].interpolate("nearest")
                # fold_val_regressors_df = fold_val_regressors_df.merge(one_hot_h_df, on="ds", how="left").fillna(0).sort_values("ds")

                model = SARIMAX(
                    endog=fold_train_df,
                    exog=fold_regressors_df.iloc[:, 1:].values,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    **hparams
                )
                m_fit = model.fit(disp=0)
                preds = m_fit.forecast(val_days, exog=fold_val_regressors_df.iloc[:, 1:].values)
                preds = preds.to_frame().reset_index().rename({"index":"ds", "predicted_mean":"yhat"}, axis=1)
                preds["ds"] = preds["ds"].dt.strftime("%Y-%m-%d")
                if hp_df is not None and (diff == "year" or diff == "both"):
                    preds = year_inv_diff([], preds, fold_prev_df)
                fold_preds_df = preds.merge(fold_val_df, on="ds")[["ds", "yhat", "y"]]

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

# def fit_predict(key_cols, train_df, test_df, stores_dict, all_holidays_df, hp_df):
#     # merge train_df with hp_df
#     train_df = train_df.merge(hp_df, on=key_cols)

#     # group by key col and apply custom function
#     def fp(x_df):
#         # get key_col values for display purposes
#         # create filter for test data
#         tup = tuple()
#         filter = pd.Series(test_df.shape[0] * [True])
#         for c in key_cols:
#             tup += (x_df[c].iloc[0], )
#             filter = filter & (test_df[c] == x_df[c].iloc[0])
        
#         # get test_data (dates and external regressors)
#         x_test_df = test_df[filter][["ds", "onpromotion", "dcoilwtico"]].reset_index(drop=True)
#         # agg x_test_df
#         x_test_df = x_test_df.groupby("ds").agg({"onpromotion":"sum", "dcoilwtico":"first"}).reset_index()

#         # get h_df
#         store_nbrs = x_df["store_nbr"].drop_duplicates()
#         states = [stores_dict[snbr]["state"] for snbr in store_nbrs]
#         cities = [stores_dict[snbr]["city"] for snbr in store_nbrs]
#         filter = (all_holidays_df["locale_name"] == "Ecuador")
#         for s in states:
#             filter = filter | (all_holidays_df["locale_name"] == s)
#         for c in cities:
#             filter = filter | (all_holidays_df["locale_name"] == c)
#         h_df = all_holidays_df[filter]
#         h_df = h_df[["ds", "holiday", "lower_window", "upper_window"]]

#         # get hyperparams
#         cp_scale = x_df["changepoint_prior_scale"].iloc[0]
#         sp_scale = x_df["seasonality_prior_scale"].iloc[0]
#         hp_scale = x_df["holidays_prior_scale"].iloc[0]
#         cr = x_df["changepoint_range"].iloc[0]

#         # aggregate x_df
#         x_df = x_df.groupby("ds").agg({"y":"sum", "onpromotion":"sum", "dcoilwtico":"first"}).reset_index()

#         # remove up to first nonzero entry of x_df
#         first_date = x_df["ds"][x_df["y"] > 0].min()
#         x_df = x_df[x_df["ds"] >= first_date]

#         if x_df.shape[0] == 0:
#             # when there is no data, return all zero predictions
#             preds = x_test_df[["ds"]]
#             preds["yhat"] = 0
#         else:
#             # instantiate model
#             model = Prophet(
#                 uncertainty_samples=0,
#                 holidays=h_df,
#                 changepoint_prior_scale=cp_scale,
#                 seasonality_prior_scale=sp_scale,
#                 holidays_prior_scale=hp_scale,
#                 changepoint_range=cr
#             )
#             model.add_regressor("onpromotion")
#             model.add_regressor("dcoilwtico")
#             # fit model
#             model.fit(x_df)
#             preds = model.predict(x_test_df)[["ds", "yhat"]]
#             preds["yhat"] = preds["yhat"].clip(0)
#             preds["ds"] = np.datetime_as_string(preds["ds"].to_numpy(), unit='D')
#         # display string
#         display_str = "{} predictions complete".format(tup)
#         print(display_str)
#         return preds

#     return train_df.groupby(key_cols)[train_df.columns].apply(fp).reset_index()[key_cols + ["ds", "yhat"]]


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

    # # subset train data for computational reasons
    # # we will optimize hyperparams over smaller subsets of the data
    # # there are 54 stores, we will do 20 at a time
    # train_df = train_df[(train_df["store_nbr"] > 40)]

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
    key_cols = ["store_nbr", "family"]
    hp_df = pd.read_csv("arima_hparams.csv").drop("Unnamed: 0", axis=1)
    msles_df = cross_validation(key_cols, train_df, stores_dict, all_holidays_df, hp_df, train_days=365*4 + 128, val_days=16, interval=28)
    msles_df.to_csv("./store_nbr_family_arima_msles.csv")
    print(msles_df.mean(axis=0)**0.5)

    # # hyperparameter optimization (coordinate descent)
    # key_cols = ["store_nbr", "family"]
    # hparam_grid = {
    #     "diff":["none", "year", "day", "both"],
    #     "order":[(8, 8), (8, 4), (4, 8), (8, 0), (0, 8), (4, 4), (4, 0), (0, 4)],
    # }
    # best_hparams_df = cv_optimize(key_cols, train_df, stores_dict, all_holidays_df, hparam_grid, train_days=365*4 + 128, val_days=16, interval=28)
    # best_hparams_df.to_csv("./arima_hparams3.csv")