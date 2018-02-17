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

def read_data(csv_file, date_col):
    df = pd.read_csv(csv_file)
    print('Read csv file: ', csv_file)

    # Convert t column to datetime and create numeric t column
    df = convert_to_datetime(df, date_col)
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

def doplot(df, result, x_col, y_col, date_col):
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

    ax1.plot(df[date_col], df[y_col], 'ok', label=y_col)
    ax1.plot(df[date_col], ynewpred, 'r-', label='Predicted')
    ax2.plot(df[date_col], residuals, 'sk')

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

def do_fit(df, x_col, y_col, date_col):
    result = run_ols( df, x_col, y_col)
    doplot(df, result, x_col, y_col, date_col)
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

def main(csv_file, date_col,y_col):
    df = read_data(csv_file, date_col)
    #df = pd.read_csv('http://vincentarelbundock.github.io/Rdatasets/csv/MASS/Boston.csv')
    print(df)
    wlog = start_weblog()
    i = 0
    with open('x_cols.dat', 'rb') as columns:
        for x_columns  in columns:
            print(x_columns)
            x_col = x_columns.strip('\n').split(',')
            result = do_fit(df, x_col, y_col, date_col)
            weblog_item(wlog, result, x_col, y_col, n=i+1)
            i += 1

if __name__ == '__main__':
    try:
        csv_file = sys.argv[1]
    except:
        csv_file = 'test_data.csv'
        print('Unable to read csv file')
        print('Usage: python mmm_fit.py <csv_file>')
    y_col = 'c3'
    date_col = 't'
    main(csv_file, date_col, y_col)


# Info:
    # https://www.datarobot.com/blog/multiple-regression-using-statsmodels/
    # https://www.datarobot.com/blog/ordinary-least-squares-in-python/

