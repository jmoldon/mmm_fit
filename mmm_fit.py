import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import leastsq
import statsmodels.api as sm
import sys

def convert_to_datetime(df, c):
    # Select a column to convert to datetime
    df[c] = pd.to_datetime(df[c])
    df[c+'_num'] = df[c].values.astype('float')
    return df

def read_data():
    try:
        csv_file = sys.argv[1]
        df = pd.read_csv(csv_file)
        print('Read csv file: ', csv_file)
    except:
        print('Unable to read csv file')
        print('Usage: python mmm_fit.py <csv_file>')
    # Convert t column to datetime and create numeric t column
    df = convert_to_datetime(df, 't')
    return df

def run_ols(df, x_col, y_col):
    # Based on http://connor-johnson.com/2014/02/18/linear-regression-with-python/
    df['intercept'] = 1.0
    x_cols = np.concatenate([np.atleast_1d(x_col), ['intercept']])
    X = df[x_cols]
    Y = df[y_col]
    result = sm.OLS( Y, X ).fit()
    print(result.summary())
    return result

def doplot():
    return


def main():
    df = read_data()
    print(df)
    result1 = run_ols( df, ['c1', 'c2'], 'c3')
    result2 = run_ols( df,'c1', 'c3')




if __name__ == '__main__':
    main()



#def regression(x, y, ey=None, p0):
#    pinit = np.ones(len(p0))
#    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
#    out = leastsq(errfunc, pinit, args=(x,y,ey), full_output=1)
#    # calculate final chi square
#    chisq = sum(out[2]["fvec"]**2.0)
#    dof   = len(x)-len(pinit)
#    rms   = np.sqrt(chisq/dof)
#    print "Chi squared        - %5.2f" %(chisq)
#    print "degrees of freedom - %5i  " %(dof)
#    print "RMS of residuals   - %5.2f" %(rms)
#    print "Reduced chisq      - %5.2f" %(rms**2.0)
#    covar = out[1]
#    p  = out[0]
#    ep = np.sqrt(asfarray([covar[i][i] for i in range(len(covar))]))
#    fit = {'p':p,
#           'ep':ep,
#           'out':out,
#           'chisq':chisq,
#           'dof':dof,
#           'rms':rms}
#    return fit


