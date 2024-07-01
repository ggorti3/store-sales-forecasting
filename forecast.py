from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from prophet import Prophet
from prophet.diagnostics import cross_validation

def fit_predict(key_cols, hp_df, stores_dict, train_df):
    pass