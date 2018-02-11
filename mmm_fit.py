import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import leastsq
import sys

def convert_to_datetime(df, c, set_index=False):
    # Select a column to convert to datetime
    # Optionally, use that column as index of the df
    df[c] = pd.to_datetime(df[c])
    df.set_index(c, inplace=True)
    return df

def read_data():
    try:
        csv_file = sys.argv[1]
        df = pd.read_csv(csv_file)
        print('Read csv file: ', csv_file)
    except:
        print('Unable to read csv file')
        print('Usage: python mmm_fit.py <csv_file>')
    # Convert t column to datetime. 
    # Could need some work if date format is not standard
    df = convert_to_datetime(df, 't', set_index=True)
    return df

def regression(x, y, ey=None, p0):
    pinit = np.ones(len(p0))
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    out = leastsq(errfunc, pinit, args=(x,y,ey), full_output=1)
    # calculate final chi square
    chisq = sum(out[2]["fvec"]**2.0)
    dof   = len(x)-len(pinit)
    rms   = np.sqrt(chisq/dof)
    print "Chi squared        - %5.2f" %(chisq)
    print "degrees of freedom - %5i  " %(dof)
    print "RMS of residuals   - %5.2f" %(rms)
    print "Reduced chisq      - %5.2f" %(rms**2.0)
    covar = out[1]
    p  = out[0]
    ep = np.sqrt(asfarray([covar[i][i] for i in range(len(covar))]))
    fit = {'p':p,
           'ep':ep,
           'out':out,
           'chisq':chisq,
           'dof':dof,
           'rms':rms}
    return fit

def doplot():
    return


def main():
    df = read_data()
    for c in df.columns:
        fit = regression()
        print(c, df[c])
    print(df)



if __name__ == '__main__':
    main()
