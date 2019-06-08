from dfply import *
import pandas as pd
import os
import datetime
from sklearn import linear_model
import time
from multiprocessing import Pool
from functools import partial
os.chdir(r"c:\Users\ChonWai\Desktop\machine learning\data")


def end_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)


def lasso(permno, msf, x_train, x_test):
    """
    perfrom a lasso regression
    :param permno:
    :param msf:
    :param x_train:
    :param x_test:
    :return: a dict which contail the following in order: predict return, alpha that use in the model, coeficient of all
    x variable
    """
    training_y = msf >> mask(X.date <= '2008-01-31', X.permno == permno) >> select(X.ret)
    testing_y = msf >> mask(X.date > '2008-01-31', X.permno == permno) >> select(X.ret)
    lasso_reg = linear_model.Lasso()
    alphas = np.logspace(-10, -1, 10)
    # try to find the best r square by trying different alpha
    r_square_list = [lasso_reg.set_params(alpha=alpha).fit(x_train, training_y).score(x_test, testing_y)
                     for alpha in alphas]
    lasso_reg.set_params(alpha=alphas[r_square_list.index(max(r_square_list))])
    lasso_reg.fit(x_train, training_y)
    return {str(permno): [lasso_reg.predict(x_test), lasso_reg.alpha, lasso_reg.coef_]}


if __name__ == '__main__':
    start_time = time.time()
    ff5 = pd.read_csv("Fama French 5 Factors.CSV", skiprows=2)
    ps = pd.read_csv("Pastor Stambaugh Factors.csv")
    hxz = pd.read_excel("HXZ q-Factors (monthly 1967 to 2014).xlsx")
    msf = pd.read_csv("cleaned_data.csv")
    # renaming the columns
    ff5.rename(columns={"Unnamed: 0": "date", "Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RMW": "rmw", "CMA": "cma",
                        "RF": "rf"}, inplace=True)

    # seperating the monthly data and the annual data from ff5 csv
    monthly_ff5 = ff5.iloc[: ff5.index[ff5.date == ' Annual Factors: January-December '].tolist()[0]]
    monthly_ff5["date"] = pd.to_datetime(monthly_ff5["date"], format='%Y%m')
    monthly_ff5.date = monthly_ff5.date.apply(end_of_month)
    monthly_ff5 = monthly_ff5.astype({"mkt_rf": float, "smb": float, "hml": float, "rmw": float, "cma": float, "rf": float})
    annual_ff5 = ff5.loc[ff5.index[ff5.date == ' Annual Factors: January-December '].tolist()[0] + 2:]
    annual_ff5.date = [date.strip() for date in annual_ff5.date]
    annual_ff5 = annual_ff5.astype({"mkt_rf": float, "smb": float, "hml": float, "rmw": float, "cma": float, "rf": float})
    hxz.rename(columns={"Year": "year", "Month": "month", "MKT": "mkt", "ME": "me", "I/A": "i/a", "ROE": "roe"},
               inplace=True)
    hxz.insert(0, "date", pd.to_datetime(hxz[["year", "month"]].assign(Day=1)))
    hxz.date = hxz.date.apply(end_of_month)
    ps.rename(columns={"DATE": "date", "PS_LEVEL": "ps_level", "PS_INNOV": "ps_innov", "PS_VWF": "ps_vwf"},
              inplace=True)
    ps.date = pd.to_datetime(ps.date, format="%Y%m%d").apply(end_of_month)
    msf.pop("index")
    msf.insert(0, "date", pd.to_datetime(msf["time_stamp"], format="%Y%m%d"))
    msf.date.apply(end_of_month)
    msf.pop("time_stamp")
    msf.rename(columns={"stock_id": "permno", "price": "prc", "return": "ret"}, inplace=True)
    all_factor = pd.merge(ps, monthly_ff5, how="outer", on="date")
    all_factor = pd.merge(all_factor, hxz, how="outer", on="date")
    # drop out irrevlent factors
    all_factor = all_factor.drop(["mkt", "year", "month"], axis=1)
    all_factor >> arrange(X.date)
    for lag_month in range(6):
        for factor in ['ps_level', 'ps_innov', 'ps_vwf', 'mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf', 'me', 'i/a',
                       'roe']:
            all_factor[factor + "_lag" + str(lag_month + 1)] = all_factor[factor].shift(lag_month + 1)
    # Lasso regression
    # seperating the training set and the testing set
    # Given that the msf start at 1989, thus the training set is began from 1989-01-31 to 2008-01-31
    # and end with 2012-12-31
    training_x = all_factor.dropna().query("date >= '1989' & date <= '2008-01-31'")[all_factor.columns[1:]]
    testing_x = all_factor.dropna().query("date > '2008-01-31' & date <= '2012-12-31' ")[all_factor.columns[1:]]
    permno_list = list(set(msf.permno))
    p = Pool(processes=7)
    result = list(p.map(partial(lasso, msf=msf, x_train=training_x, x_test=testing_x), permno_list))
    print(result)
    print("--- %s seconds ---" % (time.time() - start_time))





















