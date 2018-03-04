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
    #df[c+'_num'] = df[c].values.astype('float')  # Convert to numerical time
    return df

def standarize_df(df, date_col):
    cols = [c for c in df.columns if c != date_col]
    for c in cols:
        df[c] = (df[c]-df[c].mean())/df[c].std()
    return df

def read_data(csv_file, date_col):
    df = pd.read_csv(csv_file)
    print('Read csv file: ', csv_file)

    # Convert t column to datetime and create numeric t column
    df = convert_to_datetime(df, date_col)
    df = standarize_df(df, date_col)
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

def weblog_priority(wlog):
    wlog.write('<h1>Column priority</h1>\n')
    with open('priority.txt', 'rb') as p:
        lines = p.readlines()
        wlog.write('<table bgcolor="#eeeeee" border="3px" cellspacing = "0" cellpadding = "4px" style="width:40%">\n')
        wlog.write('<tr><td>{0:10}</td><td> {1:5}</td>\n'.format('Column', 'Rsquared improvement %'))
        for line in lines[1:]:
            c, i = line.split()
            wlog.write('<tr><td>{0:10}</td><td> {1:5}</td>\n'.format(c,i))
        wlog.write('</table><br>\n')


def weblog_item(wlog, result, x_col, y_col, n=1):
    wlog.write('<h2>Test {0}</h2>\n'.format(n))
    wlog.write('Independent columns: {}<br>\n'.format('_'.join(np.atleast_1d(x_col))))
    wlog.write('Dependent columns: {}<br><br>\n'.format('_'.join(np.atleast_1d(y_col))))
    wlog.write('Rsquared: {0:6.4f}<br><br>\n'.format(result.rsquared))
    simpletable = short_summary(result)
    richtable = replace_simpletable(simpletable)
    wlog.write(richtable)
    filename = '../plots/{0}__vs__{1}.png'.format(y_col, '_'.join(np.atleast_1d(x_col)))
    wlog.write('<a href = "{0}"><img style="max-width:700px" src="{0}"></a><br>\n'.format(filename))
    wlog.write('<hr>\n')



### Variable selection

def check_improvement_c(df, c, date_col, y_col):
    cols_no_c = [ci for ci in df.columns if ci not in [date_col, y_col, c]]
    cols      = [ci for ci in df.columns if ci not in [date_col, y_col]]
    # Without column c:
    res_wout_c = run_ols(df, x_col=cols_no_c, y_col=y_col)
    # With column c:
    res_with_c = run_ols(df, x_col=cols, y_col=y_col)
    #print res_wout_c.summary()
    #print res_with_c.summary()
    improvement =  (res_with_c.rsquared-res_wout_c.rsquared)/res_wout_c.rsquared*100.
    return improvement



def column_priority(df, date_col, y_col):
    columns = [ci for ci in df.columns if ci not in [date_col, y_col]]
    imp = []
    for c in columns:
        imp.append(check_improvement_c(df, c, date_col, y_col))
    # Short descending order:
    imp, columns = zip(*sorted(zip(imp,columns))[::-1])
    with open('priority.txt', 'wb') as p:
        p.write('{0:10} {1:5}\n'.format('Column', 'Rsquared improvement %'))
        for c, i in zip(columns, imp):
            p.write('{0:10} {1:5.3f}\n'.format(c,i))



def main(csv_file, date_col,y_col):
    df = read_data(csv_file, date_col)
    #df = pd.read_csv('http://vincentarelbundock.github.io/Rdatasets/csv/MASS/Boston.csv')
    print(df)
    # First check variable priority:
    column_priority(df, date_col, y_col)
    wlog = start_weblog()
    i = 0
    with open('x_cols.dat', 'rb') as columns:
        weblog_priority(wlog)
        for x_columns  in columns:
            print(x_columns)
            x_col = x_columns.strip('\n').split(',')
            result = do_fit(df, x_col, y_col, date_col)
            weblog_item(wlog, result, x_col, y_col, n=i+1)
            i += 1

if __name__ == '__main__':
    file_to_read = sys.argv[-3:]
    print('Usage: python mmm_fit.py <csv_file>')
    if file_to_read == 'csv':
        csv_file = file_to_read
    else:
        default_data = 'test_data.csv'
        print('Using default file: {}'.format(default_data)
        csv_file = default_data
    y_col = 'c3'
    date_col = 't'
    main(csv_file, date_col, y_col)


# Info:
    # https://www.datarobot.com/blog/multiple-regression-using-statsmodels/
    # https://www.datarobot.com/blog/ordinary-least-squares-in-python/

