from dfply import *
import pandas as pd
import os
import datetime
from sklearn import linear_model
os.chdir(r"c:\Users\ChonWai\Desktop\machine learning\data")


def end_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)


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
for lag_month in range(6):
    for factor in ['ps_level', 'ps_innov', 'ps_vwf', 'mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf', 'mkt', 'me', 'i/a',
                   'roe']:
        all_factor[factor + "_lag" + str(lag_month + 1)] = all_factor[factor].shift(lag_month + 1)
# Lasso regression
# seperating the training set and the testing set
# Given that the msf start at 1989, thus the training set is began from 1989-01-31 to 2008-01-31
# and end with 2012-12-31
training_x = monthly_ff5.query("date >= '1989' & date <= '2008-01-31'")[['mkt_rf', 'smb', 'hml', 'rmw', 'cma']]
training_y = msf.query("date <= '2008-01-31' & permno == '10001'").ret
testing_x = monthly_ff5.query("date > '2008-01-31' & date <= '2012-12-31' ")[['mkt_rf', 'smb', 'hml', 'rmw', 'cma']]
testing_y = msf.query("date > '2008-01-31' & permno == '10001'").ret
alphas = np.logspace(-4, -1, 10)
lasso_reg = linear_model.Lasso()
# try to find the best r square by trying different alpha
r_square_list = [lasso_reg.set_params(alpha=alpha).fit(training_x, training_y).score(testing_x, testing_y)
                 for alpha in alphas]
lasso_reg.aplha = alphas[r_square_list.index(max(r_square_list))]
lasso_reg.predict(testing_x)















