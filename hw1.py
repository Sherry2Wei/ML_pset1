from dfply import *
import pandas as pd
import os
import datetime
from sklearn import linear_model
import time
from multiprocessing import Pool
from functools import partial
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from collections import Counter
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
    in_sample_mse = mean_squared_error(training_y, lasso_reg.predict(x_train))
    out_sample_mse = mean_squared_error(testing_y, lasso_reg.predict(x_test))
    return {"permno": str(permno), "pred_y": lasso_reg.predict(x_test), "alpha": lasso_reg.alpha,
            "coef": lasso_reg.coef_, "in_sample_mse": in_sample_mse, "out_sample_mse": out_sample_mse}


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
    result = list(map(partial(lasso, msf=msf, x_train=training_x, x_test=testing_x), permno_list))
    p.close()
    pred_y_list = []
    alpha_list = []
    coef_list = []
    in_sample_error_list = []
    out_sample_error_list = []
    for dictionary in result:
        pred_y_list.append(dictionary["pred_y"].tolist())
        alpha_list.append(dictionary["alpha"])
        coef_list.append(dictionary["coef"].tolist())
        in_sample_error_list.append(dictionary["in_sample_mse"])
        out_sample_error_list.append(dictionary["out_sample_mse"])
    x = [str(permno) for permno in permno_list]
    y_in_sample = in_sample_error_list
    y_out_sample = out_sample_error_list
    ax1 = plt.subplot(1, 1, 1)
    postion_x = np.arange(len(permno_list))
    width = 0.3
    plt.xticks(postion_x + width / 2, x, rotation='vertical')
    in_sample_bar = ax1.bar(postion_x, y_in_sample, width=0.3, color="blue")
    ax2 = ax1.twinx()
    out_sample_bar = ax2.bar(postion_x + 0.3, y_out_sample, width=0.3, color="red")
    plt.legend([in_sample_bar, out_sample_bar], ["In Sample MSE", "Out Sample MSE"])
    plt.ylabel("MSE", fontdict={"fontweight": "bold"})
    plt.xlabel("Permno", fontdict={"fontweight": "bold"})
    plt.title("In Sample and out sample MSE comparsion",
              fontdict={"fontstyle": "italic", "fontweight": "bold"})
    plt.show()
    plt.close()
    # The in sample and out of sample prediction error
    print("average in sample error of all assets:")
    print(np.average(y_in_sample))
    print("average out sample error of all assets:")
    print(np.average(y_out_sample))
    # plot the factors selection frequency for all sample
    factors = []
    for permno_coef in coef_list:
        permno_factor = [all_factor.columns[1:][permno_coef.index(coef)] for coef in permno_coef if coef > 0]
        factors.extend(permno_factor)
    factors_freq = Counter(factors)
    factors_freq = pd.Series(factors_freq)
    x_factor = factors_freq.index.tolist()
    y_factor_freq = factors_freq.values.tolist()
    y_factor_freq.sort(reverse=True)
    postion_x_factor = np.arange(len(x_factor))
    width = 0.3
    plt.xticks(postion_x_factor, x_factor, rotation='vertical')
    freq_bar = plt.bar(postion_x_factor, y_factor_freq, width=0.3, color="blue")
    plt.legend(freq_bar, ["Factors Frequency"])
    plt.ylabel("Frequency", fontdict={"fontweight": "bold"})
    plt.xlabel("Factors", fontdict={"fontweight": "bold"})
    plt.title("Factors Selection Frequency ",
              fontdict={"fontstyle": "italic", "fontweight": "bold"})
    plt.show()
    plt.close()
    print("--- %s seconds ---" % (time.time() - start_time))





















