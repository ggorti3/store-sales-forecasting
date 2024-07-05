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

import pandas as pd

def get_recent_sales_covars(covar_days, val_days):
    # group by store_nbr and family then apply custom function?

    # data loading and processing
    train_df = pd.read_csv("./train.csv")
    train_df = train_df.rename({"date":"ds", "sales":"y"}, axis=1)

    # subset for reasonable sized calculations
    train_df = train_df[(train_df["store_nbr"] < 11)]

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
    blank_oil_df = pd.DataFrame({"ds":pd.date_range(train_df["ds"].min(), test_df["ds"].max()).astype("str")})
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


if __name__ == "__main__":
    train_df = get_recent_sales_covars(covar_days=21, val_days=16)
    train_df = add_external_covars(train_df)
    train_df = add_holidays_covars(train_df)
    train_df.to_csv("./global_train_1.csv")

