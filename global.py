# fit one global model to "forecast" at the store_nbr, family level
# format problem as regression, using the following covariates:
# - last few sales values of that timeseries
# - month indicators
# - weekday indicators
# - days from most recent provided timeseries value (days_out)
# - oil price at prediction time
# - holiday indicators
# - cluster
# - store_nbr indicators
# - family indicators

import numpy as np
import pandas as pd
import lightgbm as lgb
import re
from datetime import date, timedelta
from dataset_h5py import save2hdf
import h5py
import os

FAMILY_CODES = {
    'AUTOMOTIVE': 0,
    'BABY CARE': 1,
    'BEAUTY': 2,
    'BEVERAGES': 3,
    'BOOKS': 4,
    'BREAD/BAKERY': 5,
    'CELEBRATION': 6,
    'CLEANING': 7,
    'DAIRY': 8,
    'DELI': 9,
    'EGGS': 10,
    'FROZEN FOODS': 11,
    'GROCERY I': 12,
    'GROCERY II': 13,
    'HARDWARE': 14,
    'HOME AND KITCHEN I': 15,
    'HOME AND KITCHEN II': 16,
    'HOME APPLIANCES': 17,
    'HOME CARE': 18,
    'LADIESWEAR': 19,
    'LAWN AND GARDEN': 20,
    'LINGERIE': 21,
    'LIQUOR,WINE,BEER': 22,
    'MAGAZINES': 23,
    'MEATS': 24,
    'PERSONAL CARE': 25,
    'PET SUPPLIES': 26,
    'PLAYERS AND ELECTRONICS': 27,
    'POULTRY': 28,
    'PREPARED FOODS': 29,
    'PRODUCE': 30,
    'SCHOOL AND OFFICE SUPPLIES': 31,
    'SEAFOOD': 32
}

MONTH_CODES = {
    "January":0,
    "February":1,
    "March":2,
    "April":3,
    "May":4,
    "June":5,
    "July":6,
    "August":7,
    "September":8,
    "October":9,
    "November":10,
    "December":11
}

DAY_CODES = {
    "Sunday":0,
    "Monday":1,
    "Tuesday":2,
    "Wednesday":3,
    "Thursday":4,
    "Friday":5,
    "Saturday":6
}

def get_recent_sales_covars(covar_days, val_days):
    # group by store_nbr and family then apply custom function?

    # data loading and processing
    train_df = pd.read_csv("./train.csv")
    train_df = train_df.rename({"date":"ds", "sales":"y"}, axis=1)

    # subset for reasonable sized calculations
    train_df = train_df[(train_df["store_nbr"] >= 21) & (train_df["store_nbr"] < 31)]

    def expand_days(x_df):
        n = x_df.shape[0]
        x_df = x_df.sort_values("ds")
        print("store_nbr: {}, family: {}".format(x_df["store_nbr"].iloc[0], x_df["family"].iloc[0]))

        covar_slice_dfs = []
        days_outs = []
        ys = []
        ds_list = []
        for vdate_idx in range(covar_days + 1, n):
            days_out = 1
            while vdate_idx - days_out >= covar_days and days_out <= val_days:

                covar_slice = x_df["y"].iloc[vdate_idx - covar_days - days_out + 1:vdate_idx - days_out + 1]
                covar_slice_dict = {"x_{}".format(i + 1):[covar_slice.iloc[i]] for i in range(covar_days)}
                covar_slice_df = pd.DataFrame(covar_slice_dict)
                covar_slice_dfs.append(covar_slice_df)

                days_outs.append(days_out)
                ys.append(x_df["y"].iloc[vdate_idx])
                ds_list.append(x_df["ds"].iloc[vdate_idx])
                days_out += 1
        covar_df = pd.concat(covar_slice_dfs, axis=0)
        covar_df["ds"] = ds_list
        covar_df["days_out"] = days_outs
        covar_df["y"] = ys
        return covar_df
    
    return train_df.groupby(["store_nbr", "family"])[train_df.columns].apply(expand_days).reset_index().drop("level_2", axis=1)

def add_holidays_covars(train_df):
    assert "store_nbr" in train_df.columns, "train_df must have column 'store_nbr'"
    assert "ds" in train_df.columns, "train_df must have column 'ds'"

    # add locale_name to train_df using store_nbr
    stores_df = pd.read_csv("stores.csv")
    train_df = train_df.merge(
        stores_df[["store_nbr", "city", "state"]],
        on="store_nbr",
        how="left"
    )

    # read holidays_events.csv
    holidays_df = pd.read_csv("holidays_events.csv")

    # resolve transferred holidays
    nth_df = holidays_df[(holidays_df["transferred"] == False) & (holidays_df["type"] != "Transfer")]
    th_df = holidays_df[holidays_df["type"] == "Transfer"]
    th_df["description"] = th_df["description"].str.removeprefix("Traslado ")
    all_holidays_df = pd.concat([nth_df, th_df], axis=0)[["date", "locale_name", "description"]]

    # rename columns
    all_holidays_df = all_holidays_df.rename({"date":"ds", "description":"holiday"}, axis=1).drop_duplicates()
    all_holidays_df["dummy"] = 1

    # get regional holidays
    regional_holidays_df = all_holidays_df[all_holidays_df["locale_name"] != "Ecuador"].reset_index(drop=True)
    # remove -1 holidays
    regional_holidays_df = regional_holidays_df[~regional_holidays_df["holiday"].str.endswith("-1")].reset_index(drop=True)
    # take prefixes only
    regional_holidays_df["holiday"] = regional_holidays_df["holiday"].str.split(expand=True).loc[:, 0]
    # pivot regional holidays on locale_name and ds (rename locale_name to region)
    one_hot_regional_holidays_df = regional_holidays_df.pivot(index=["ds", "locale_name"], columns="holiday", values="dummy").reset_index()

    # left merge train_df with one_hot_regional_holidays_df on ds and city
    train_df_city = train_df.merge(
        one_hot_regional_holidays_df,
        how="left",
        left_on=["city", "ds"],
        right_on=["locale_name", "ds"]
    ).drop(
        ["city", "state", "locale_name"],
        axis=1
    ).fillna(0)
    # left merge train_df with one_hot_regional_holidays on ds and state
    train_df_state = train_df.merge(
        one_hot_regional_holidays_df,
        how="left",
        left_on=["state", "ds"],
        right_on=["locale_name", "ds"]
    ).drop(
        ["city", "state", "locale_name"],
        axis=1
    ).fillna(0)

    # take max of one hot columns to get all holidays celebrated that day in both city and state
    train_df = train_df_state.where(train_df_city <= train_df_state, train_df_city)


    # get national holidays
    national_holidays_df = all_holidays_df[all_holidays_df["locale_name"] == "Ecuador"].reset_index(drop=True)
    # pivot national holidays on ds
    one_hot_national_holidays_df = national_holidays_df.pivot(index="ds", columns="holiday", values="dummy").reset_index()
    # left merge train_df with one_hot_national_holidays_df on ds
    train_df = train_df.merge(one_hot_national_holidays_df, how="left", on="ds").fillna(0)
    

    return train_df



def add_external_covars(train_df):
    assert "store_nbr" in train_df.columns, "train_df must have column 'store_nbr'"
    assert "ds" in train_df.columns, "train_df must have column 'ds'"
    # oil, cluster, month and day

    # load oil price data
    oil_df = pd.read_csv("oil.csv")
    oil_df = oil_df.rename({"date":"ds"}, axis=1)
    # interpolate missing oil prices
    blank_oil_df = pd.DataFrame({"ds":pd.date_range(train_df["ds"].min(), train_df["ds"].max()).astype("str")})
    oil_df = blank_oil_df.merge(oil_df, how="left", on="ds")
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].interpolate("nearest")
    oil_df.iloc[0, 1] = 93.14
    # add oil prices
    train_df = train_df.merge(oil_df, how="left", on="ds")

    # get stores data
    stores_df = pd.read_csv("stores.csv")
    # add cluster
    train_df = train_df.merge(stores_df[["store_nbr", "cluster"]], how="left", on="store_nbr")

    # month and day (keep as strings for easy one-hot-encoding by the light gbm)
    train_df["month"] = pd.to_datetime(train_df["ds"]).dt.month_name()
    train_df["day"] = pd.to_datetime(train_df["ds"]).dt.day_name()

    return train_df

class HDFSequence(lgb.Sequence):

    def __init__(self, hdf_fname, batch_size):
        self.data = h5py.File(hdf_fname, "r")["train_X"]
        self.batch_size = batch_size

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def rmsle_eval(preds, eval_data):
    labels = eval_data.get_label()
    return "rmsle", np.sqrt(np.mean((np.log(1 + preds) - np.log(1 + labels))**2)), False


def theplan(file_list, d_split, batch_size):
    # take in list of all csv files
    # for each file

    # list of train labels and list of val labels
    train_y_list = []
    val_y_list = []
    # list of hdf file
    hdf_file_list = []
    # list of val_X
    val_X_list = []
    for i, file in enumerate(file_list):
        # load df from csv
        df = pd.read_csv(file)

        # convert string columns to int for lbgm
        df["family"] = df["family"].apply(lambda x: FAMILY_CODES[x])
        df["month"] = df["month"].apply(lambda x: MONTH_CODES[x])
        df["day"] = df["day"].apply(lambda x: DAY_CODES[x])
        # drop non variable columns
        stupid_cols = []
        for c in df.columns:
            if c.startswith("Unnamed"):
                stupid_cols.append(c)
        df = df.drop(stupid_cols, axis=1)

        # split into train_X, train_y and val_X, val_y using d_split
        train = df[df["ds"] < d_split]
        train_X = train.drop(["ds", "y"], axis=1)
        train_y_list.append(train["y"].values)

        if i == 0:
            # rename cols to lgb friendly names
            train_X = train_X.rename(columns = lambda x:re.sub('[+]+', 'plus', x))
            train_X = train_X.rename(columns = lambda x:re.sub('[-]+', 'minus', x))
            train_X = train_X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            names = train_X.columns
            names = [n for n in train_X.columns]

        val_filter = (pd.to_datetime(df["ds"]) - pd.to_timedelta(df["days_out"] - 1, "d")).dt.strftime("%Y-%m-%d") == d_split
        val = df[val_filter]
        val_X = val.drop(["ds", "y"], axis=1)
        val_X_list.append(val_X.values)
        val_y_list.append(val["y"].values)

        # save large train data into hdf file
        fname = file.split(".")[0] + ".hdf"
        save2hdf({"train_X":train_X.values}, fname=fname, batch_size=batch_size)
        hdf_file_list.append(fname)

    train_y = np.concatenate(train_y_list)
    val_X = np.concatenate(val_X_list)
    val_y = np.concatenate(val_y_list)

    # create lgb train dataset using sequences
    train_seq_list = [HDFSequence(hdf_fname, batch_size) for hdf_fname in hdf_file_list]
    train_dataset = lgb.Dataset(
        train_seq_list,
        feature_name=names,
        categorical_feature=["store_nbr", "family", "cluster", "month", "day"],
        label=train_y
    )

    # create validation set
    # val_dataset = train_dataset.create_valid(val_X, label=val_y)
    val_dataset = lgb.Dataset(
        val_X,
        feature_name=names,
        categorical_feature=["store_nbr", "family", "cluster", "month", "day"],
        label=val_y,
        reference=train_dataset
    )

    # save datasets
    train_dataset.save_binary("./train.bin")
    val_dataset.save_binary("./val.bin")

    # param = {'num_leaves': 31, 'objective':'regression'}#, "metric":"None"}
    # num_round = 10
    # bst = lgb.train(param, train_dataset, num_round, valid_sets=[val_dataset])#, feval=rmsle_eval)
    # bst.add_valid(val_dataset, "myvalset")
    # val_result = bst.eval_valid()#feval=rmsle_eval)
    # bst.save_model('model.txt')
    # print(val_result)

    # delete h5py files
    for f in hdf_file_list:
        os.remove(f)
    
    # return train_dataset, val_dataset


if __name__ == "__main__":
    # # converting raw data to global model data
    # train_df = get_recent_sales_covars(covar_days=21, val_days=16)
    # train_df = add_external_covars(train_df)
    # train_df = add_holidays_covars(train_df)
    # train_df.to_csv("./global_train_3.csv")

    # # sort data by ds
    # train_df = pd.read_csv("global_train_3.csv")
    # train_df = train_df.sort_values("ds")
    # train_df.to_csv("./global_train_3.csv", mode="w")

    theplan(
        ["global_train_1.csv", "global_train_2.csv"],
        d_split="2016-08-01",
        batch_size=33*16
    )

