import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import sys, os

fig_width = 8
fig_height = 8

def makedir(pathdir):
    try:
        os.mkdir(pathdir)
        print('Create directory: {}'.format(pathdir))
    except:
        #print('Cannot create directory: {}'.format(pathdir))
        pass


def convert_to_datetime(df, c):
    # Select a column to convert to datetime
    df[c] = pd.to_datetime(df[c])
    df[c+'_num'] = df[c].values.astype('float')
    return df

def read_data(csv_file=None):
    if csv_file == None:
        try:
            csv_file = sys.argv[1]
        except:
            print('Unable to read csv file')
            print('Usage: python mmm_fit.py <csv_file>')
    df = pd.read_csv(csv_file)
    print('Read csv file: ', csv_file)

    # Convert t column to datetime and create numeric t column
    df = convert_to_datetime(df, 't')
    return df

def run_ols(df, x_col, y_col):
    #http://www.statsmodels.org/dev/examples/notebooks/generated/predict.html
    X = df[x_col]
    X = sm.add_constant(X)
    y = df[y_col]
    olsmod = sm.OLS(y, X)
    olsres = olsmod.fit()
    #print(olsres.summary())
    return olsres

def doplot(df, result, x_col, y_col):
    makedir('./plots')
    print result.summary()
    ynewpred =  result.predict() # predict out of sample
    residuals = df[y_col]-ynewpred
    
    fig = plt.figure(figsize = (fig_width, fig_width))
    
    # definitions for the axes
    left_1, width_1 = 0.1, 0.8
    left_2, width_2 = 0.1, 0.8
    bottom_1, height_1 = 0.3, 0.65
    bottom_2, height_2 = 0.1, 0.2
    rect1 = [left_1, bottom_1, width_1, height_1]
    rect2 = [left_2, bottom_2, width_2, height_2]

    ax1 = fig.add_axes(rect1)
    ax2 = fig.add_axes(rect2)
    
    ax1.plot(df['t'], df[y_col], 'ok', label=y_col)
    ax1.plot(df['t'], ynewpred, 'r-', label='Predicted')
    ax2.plot(df['t'], residuals, 'sk')
    
    ax2.axhline(0.0, 0.0, 1.0, color ='k', ls = '--', linewidth = 0.8)
    lim_res = np.max(np.abs([ax2.get_ybound()]))
    ax2.set_ylim(-lim_res*3, lim_res*3)
    
    ax1.set_xticklabels([])
    ax1.legend(loc=0)
    
    ax1.set_ylabel('value ({})'.format(y_col))
    ax1.set_xlabel('Time')
    ax2.set_ylabel('Residuals')
    
    filename = '{0}__vs__{1}.png'.format(y_col, '_'.join(np.atleast_1d(x_col)))
    ax1.set_title(filename)
    
    fig.savefig('./plots/'+filename)
    return

def do_fit(df, x_col, y_col):
    result = run_ols( df, x_col, y_col)
    doplot(df, result, x_col, y_col)
    return result
    
    
def short_summary(est):
    return est.summary().tables[1].as_html()
    
def start_weblog():
    makedir('./weblog')
    wlog = open("./weblog/index.html","w")
    return wlog
    
def replace_simpletable(simpletable):
    richtable = simpletable.replace('class="simpletable"', 'bgcolor="#eeeeee" border="3px" cellspacing = "0" cellpadding = "4px" style="width:30%"')
    return richtable
    
def weblog_item(wlog, result, x_col, y_col, n=1):
    wlog.write('<h1>Test {0}</h1>\n'.format(n))
    wlog.write('Independent columns: {}<br>\n'.format('_'.join(np.atleast_1d(x_col))))
    wlog.write('Dependent columns: {}<br><br>\n'.format('_'.join(np.atleast_1d(y_col))))
    simpletable = short_summary(result)
    richtable = replace_simpletable(simpletable)
    wlog.write(richtable)
    filename = '../plots/{0}__vs__{1}.png'.format(y_col, '_'.join(np.atleast_1d(x_col)))
    wlog.write('<a href = "{0}"><img style="max-width:700px" src="{0}"></a><br>\n'.format(filename))
    wlog.write('<hr>\n')

def main():
    df = read_data()
    print(df)
    y_col = 'c3'
    wlog = start_weblog()
    i = 0
    with open('x_cols.dat', 'rb') as columns:
        for x_columns  in columns:
            print(x_columns)
            x_col = x_columns.strip('\n').split(',')
            result = do_fit(df, x_col, y_col)
            weblog_item(wlog, result, x_col, y_col, n=i+1)
            i += 1

if __name__ == '__main__':
    main()


# Info:
    # https://www.datarobot.com/blog/multiple-regression-using-statsmodels/
    # https://www.datarobot.com/blog/ordinary-least-squares-in-python/

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


