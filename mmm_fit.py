import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def read_data():
    try:
        csv_file = sys.argv[1]
        df = pd.read_csv(csv_file)
        print('Read csv file: ', csv_file)
    except:
        print('Unable to read csv file')
        print('Usage: python mmm_fit.py <csv_file>')
    return df

def regression():
    return

def doplot():
    return


def main():
    df = read_data()




if __name__ == '__main__':
    main()
