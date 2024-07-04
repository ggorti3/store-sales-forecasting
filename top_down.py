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

def year_diff(key_cols, support_cols, prop_df):
    other_df = prop_df.copy()
    other_df["ds"] = (pd.to_datetime(other_df["ds"]) + DateOffset(years=1)).dt.strftime("%Y-%m-%d")
    other_df = other_df.groupby(key_cols + support_cols + ["ds"]).agg("first").reset_index()
    other_df = other_df.rename({"prop":"prev_prop"}, axis=1)[key_cols + support_cols + ["ds", "prev_prop"]]
    leap_yrs_df = other_df[other_df["ds"] == "2016-02-28"]
    leap_yrs_df["ds"] = "2016-02-29"
    other_df = pd.concat([other_df, leap_yrs_df]).sort_values(key_cols + support_cols + ["ds"]).reset_index(drop=True)
    dprop_df = prop_df.merge(other_df, how="inner", on=key_cols + support_cols + ["ds"])
    dprop_df["prop"] = dprop_df["prop"] - dprop_df["prev_prop"]
    dprop_df = dprop_df[key_cols + support_cols + ["ds", "prop", "onpromotion", "dcoilwtico"]]
    return dprop_df.sort_values(key_cols + support_cols + ["ds"]), other_df.sort_values(key_cols + support_cols + ["ds"])

def year_inv_diff(key_cols, support_cols, prop_hat_df, other_df):
    prop_hat_df = prop_hat_df.merge(other_df, how="inner", on=key_cols + support_cols + ["ds"])
    prop_hat_df["prop_hat"] = prop_hat_df["prop_hat"] + prop_hat_df["prev_prop"]
    prop_hat_df = prop_hat_df[key_cols + support_cols + ["ds", "prop_hat"]]
    return prop_hat_df

class VARModel():
    def __init__(self, lag, support_cols):
        self.lag = lag
        self.support_cols = support_cols
    
    def fit(self, x_df, lmbda):
        px_df = x_df.pivot(columns=self.support_cols, index="ds", values="prop").sort_index()
        oil_df = x_df[["ds", "dcoilwtico"]].drop_duplicates().sort_values("ds")
        # need to use loop to build design matrix
        design_cols = []
        for l in range(self.lag):
            dc = px_df.iloc[l:(px_df.shape[0] - self.lag + l), :].values.flatten()
            design_cols.append(dc)
        design_cols.append(np.repeat(oil_df["dcoilwtico"].values[(self.lag):], px_df.shape[1]))
        X = np.stack(design_cols, axis=1)
        y = px_df.iloc[self.lag:, :].values.flatten()
        self.beta = lin_reg(X, y, lmbda)

        self.px = px_df.values[-self.lag:, :].T
        self.d = datetime.strptime(px_df.index[-1], "%Y-%m-%d").date() + timedelta(days=1)
        self.support = px_df.columns
    
    def predict(self, test_oil_df, days):
        test_oil_df = test_oil_df[["ds", "dcoilwtico"]].drop_duplicates().set_index("ds").sort_index()
        ox = np.full((self.px.shape[0], 1), test_oil_df.loc[self.d.strftime("%Y-%m-%d"), "dcoilwtico"])
        bx = np.ones((self.px.shape[0], 1))
        x = np.concatenate([bx, self.px, ox], axis=1)
        d0 = self.d
        out = []
        for i in range(days):
            if i > 0:
                self.px = np.concatenate([self.px[:, 1:], y[:, np.newaxis]], axis=1)
                self.d = self.d + timedelta(days=1)
                ox = np.full((self.px.shape[0], 1), test_oil_df.loc[self.d.strftime("%Y-%m-%d"), "dcoilwtico"])
                bx = np.ones((self.px.shape[0], 1))
                x = np.concatenate([bx, self.px, ox], axis=1)
                
            y = x @ self.beta
            out.append(y)
        ds = pd.date_range(start=d0.strftime("%Y-%m-%d"), periods=days, freq="D", inclusive="left").repeat(self.px.shape[0]).strftime("%Y-%m-%d")
        pdf_dict = {"ds":ds}
        if len(self.support_cols) == 1:
            pdf_dict[self.support_cols[0]] = np.tile(self.support.to_numpy(), (days, ))
        else:
            for j, sc in enumerate(self.support_cols):
                pdf_dict[sc] = np.tile(np.array([self.support[i][j] for i in range(len(self.support))]), (days, ))
            pass
        pdf_dict["prop_hat"] = np.concatenate(out)
        preds_df = pd.DataFrame(pdf_dict)
        return preds_df


def lin_reg(X, y, lmbda):
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    beta = np.linalg.solve(X.T @ X + lmbda * np.eye(X.shape[1]), X.T @ y)
    return beta


def cross_validation(key_cols, support_cols, hp_df, train_df, stores_dict, all_holidays_df, train_days=365*3, val_days=16, interval=64):
    # aggregate train_df to lowest level to get truths
    thresh = 1e-6
    truth_df = train_df.groupby(key_cols + support_cols + ["ds"]).agg({"y":"sum"}).reset_index().sort_values(key_cols + support_cols + ["ds"])
    # generate prop_df from temp_df
    # get proportions and transform
    train_prop_df = get_proportions(key_cols, support_cols, train_df)
    # # year diff transform
    # train_prop_df, train_prev_prop_df = year_diff(key_cols, support_cols, train_prop_df)

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
        x_pred_df = model.predict(x_df[val_filter][["ds", "onpromotion", "dcoilwtico"]])[["ds", "yhat"]]
        # clip yhat
        x_pred_df["yhat"] = x_pred_df["yhat"].clip(0)
        x_pred_df["ds"] = np.datetime_as_string(x_pred_df["ds"].to_numpy(), unit='D')
        # add truth values
        x_pred_df = x_pred_df.merge(x_df[val_filter][["ds", "y"]], how="inner", on=["ds"])

        return x_pred_df

    def fit_predict_VAR(x_df, d1, d2, d3):
        # assume x_df is prop_df

        # get train and val filters
        train_filter = (x_df["ds"] >= d1) & (x_df["ds"] < d2)
        val_filter = (x_df["ds"] >= d2) & (x_df["ds"] < d3)

        # instantiate model
        model = VARModel(lag=21, support_cols=support_cols)
        # fit model on train data
        model.fit(x_df[train_filter], 1e-4)

        # get val dates
        days = x_df[val_filter]["ds"].drop_duplicates().shape[0]
        # return predictions over val dates
        return model.predict(x_df[val_filter], days)

    # merge train_df with hp_df
    train_df = train_df.merge(hp_df, how="left", on=key_cols)
    # we will wait for the model training apply call to aggregate train data and generate prop data, respectively

    # initialize earliest date
    d0 = datetime.strptime(train_df["ds"].iloc[0], "%Y-%m-%d").date() + timedelta(days=365)
    d_max = datetime.strptime(train_df["ds"].iloc[-1], "%Y-%m-%d").date()

    # cv loop
    d = d0
    errs = []
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

        # fit var model at key_col + support_col level and generate prop predictions
        prop_preds_df = train_prop_df.groupby(key_cols)[train_prop_df.columns].apply(lambda x: fit_predict_VAR(x, d1, d2, d3)).reset_index()
        # # inverse diff transform prop predictions
        # prop_preds_df = year_inv_diff(key_cols, support_cols, prop_preds_df, train_prev_prop_df)

        # merge prophet and var predictions at key_col level and multiply to calculate lowest level predictions
        pred_df = prop_preds_df.merge(key_level_pred_df, on=key_cols + ["ds"], how="left")
        pred_df["yhat"] = (pred_df["yhat"] * pred_df["prop_hat"])
        pred_df = pred_df.drop("prop_hat", axis=1)
        pred_df["yhat"] = pred_df["yhat"].clip(lower=0)

        # merge predictions with truth and calculate rmsle
        pred_df = pred_df.merge(truth_df, how="inner", on=key_cols + support_cols + ["ds"])
        err = msle(pred_df)**0.5
        print("{} level error: {}".format(key_cols + support_cols, err))

        d += timedelta(days=interval)

    pass

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

    ## append cluster
    train_df = train_df.merge(stores_df[["store_nbr", "cluster"]], how="left", on="store_nbr")
    test_df = test_df.merge(stores_df[["store_nbr", "cluster"]], how="left", on="store_nbr")

    # # subset train data for debugging
    # subset_families = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY']
    # train_df = train_df[(train_df["cluster"] < 4) & (train_df["family"].isin(subset_families))]


    # setup
    key_cols = ["cluster", "family"]
    support_cols = ["store_nbr"]

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

    cross_validation(key_cols, support_cols, cf_hp_df, train_df, stores_dict, all_holidays_df, train_days=365*3, val_days=16, interval=64)