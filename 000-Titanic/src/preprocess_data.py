#!/usr/bin/env python3

"""
Script that preprocess data for the forecasting the value of BTC:
"""

from datetime import datetime
import numpy as np
# import tensorflow as tf
# import pandas as pd
import os
from sklearn import preprocessing  # pip install sklearn
from collections import deque
import random
import zipfile


def clean_screen():
    """
    [Function that just clean the screen]

    Args:
        None

    Returns:
        Nothing
    """

    from os import system, name
    _ = system('cls') if name == 'nt' else system('clear')


def read_name():
    """
    [Function that reads (as input) the name(string) of the dataset]

    Args:
        None

    Returns:
        The first location of the file or None otherwise
    """

    import glob
    import typer
    from subprocess import check_output

    print("Files in current folder")
    print(check_output(["ls", "../Titanic"]).decode("utf8"))

    print("Type 'x' to exit")
    files = []
    while (len(files) == 0):
        full_name = input("Name of the Dataset - zip file: ")
        # debugging = input("- Debugging mode [y/n]?: ")
        # modeltype = input("Type the name of the Dataset - zip file: ")

        # data_path: Input data should not be a constant since the repo should
        # be general.

        if (full_name == 'x' or full_name == 'X'):
            return None
        zip_name = full_name.casefold().split(".")[0] + ".zip"

        files = glob.glob("." + '/**/' + zip_name, recursive=True)
        print()
        if (len(files) != 0):
            print("0. File found at: ", files[0])
            return files[0]

        else:
            print("File not found")


def is_csv(a_str):
    """
    [Function that finds if a string ends in the letters 'csv' or 'CSV']

    Args:
        a_str ([str]): [A none empty string]
    """

    return ((a_str.split(".")[1] == 'csv') or (a_str.split(".")[1] == 'CSV'))


def extract_fromzip(a_zipfile):
    import sys

    """
    [Function that extracts all compressed files from a zip archive]

    Args:
        a_zipfile ([type]): [a valid zip file]

    Returns:
        None if fails or the decompressed files in current path
    """

    from zipfile import ZipFile

    # size of the file in MB
    size_MB = os.path.getsize(a_zipfile) / 1000000

    if (size_MB == 0.0):
        print("Empty file")
        sys.exit(0)

    print()
    print("1. Decompressing: {:>24}".format(a_zipfile.split("/")[1]), end="")
    print("\b\tSize = {} MB".format(size_MB))
    print()
    try:
        with ZipFile(a_zipfile, 'r') as zipObj:
            # Get a list of all archived file names from the zip
            listOfFileNames = zipObj.namelist()

            count_csv_files = sum(is_csv(afile) for afile in listOfFileNames)
            # No CSV files in ZIP file
            if (count_csv_files == 0):
                print("\tNo CSV files in {}".format(a_zipfile))
                return None

            # Iterate over each file and extract it
            for i, fileName in enumerate(listOfFileNames):
                zipObj.extract(fileName, '.')
                size_MB = os.path.getsize(fileName) / 1000000

                print("\tExtracting:  {:>21}".format(fileName), end="")
                print("\b\tSize = {} MB".format(size_MB))

            print("\n\tTotal CSV files extracted: {}".format(i + 1))
            print()
            return zipObj.namelist()

    except BaseException as e:
        print(e)
        return None


def save_csv_list(lst, filename):
    """[save list in a csv_files.yaml file]

    Args:
        lst         ([lst]):    [list of filenames]
        filename    ([str]):    [Name of the file]
    """
    import yaml

    # save to a file
    print("2. Serializing file list\n")
    with open(filename + '.yaml', "w") as fh:
        yaml.dump_all(lst, fh, explicit_start=True, default_flow_style=False)


def recover_name(a_str, a_lst):
    """[Recover the fisrt occurrence of a substring in a list]

    Args:
        a_str   ([str]):    [str to be found in any element of the list]
        a_lst   ([lst]):    [list of strings]

    """
    matching = None
    matching = [s for s in a_lst if a_str in s]
    return matching[0]


def read_csv(a_str):
    """
    [Function that read a csv file can convert the data into a panda Df]

    Args:
        a_str ([str]):        [str that reprents the name of the file to open]

    Returns:
        df         ([Pandas DF]):   [A Pandas dataframe of the open file]
    """

    import pandas as pd
    import glob

    csv_file = a_str + ".csv"
    files = glob.glob("." + '/**/' + csv_file, recursive=True)

    size_MB = os.path.getsize(csv_file) / 1000000
    print("\tOpening {} as Pandas DF".format(csv_file), end="\t")
    print("\b Size = {} MB".format(size_MB))
    df = pd.read_csv("./" + csv_file, error_bad_lines=False)

    return df


def preprocess(full_df, NAME):
    """
    [Function that preprocess data from a Pandas Df]

    Args:
        full_df ([Pandas df]):  [Dataframe with the trading info of BTC]
        NAME     ([str]):       [Name of the dataframe]

    Returns:
        main_df ([Pandas df]):  [A sliced preprocessed version of the DF]
    """
    """
    print("6. Preprocessing data of {}".format(NAME))

    # a. Converting Timestamps
    print("6.a. Converting Timestamps from UNIX to UTC")
    full_df['Timestamp'] = pd.to_datetime(full_df['Timestamp'], unit='s')

    # b. Filling the GAPs
    print("6.b. Filling the gaps. Interpolating NAN with last known value")
    full_df.fillna(method='pad', inplace=True)
    # main_df.fillna(method="ffill", inplace=True)

    # c. Slicing data
    print("6.c. Slicing data:")
    init_year = 2017

    print("\tDataframe current shape = {}".format(full_df.shape))
    print("\t\t- Removing data older than {}".format(init_year))

    full_df["year"] = pd.DatetimeIndex(full_df["Timestamp"]).year
    full_df = full_df[full_df["year"] >= init_year]
    sliced_df = full_df.drop(["year"], axis=1)

    print("\t\t- Subsamplig data to only use data each 60 min intervals")
    sliced_df = sliced_df[::60]  # start, stop, step.

    print("\t\t- Removing 'Open' column")
    sliced_df.drop(["Open"], axis=1, inplace=True)

    print("\t\t- Removing 'High' column")
    sliced_df.drop(["High"], axis=1, inplace=True)

    print("\t\t- Removing 'Low' column")
    sliced_df.drop(["Low"], axis=1, inplace=True)

    print("\t\t- Removing 'Volume_(BTC)' column")
    sliced_df.drop(["Volume_(BTC)"], axis=1, inplace=True)

    print("\t\t- Removing 'Weighted_Price' column")
    sliced_df.drop(["Weighted_Price"], axis=1, inplace=True)

    new_col_names = {'Close': 'Close_USD', 'Volume_(Currency)': 'Vol_USD'}

    print("\t\t- Renaming remaining columns")
    sliced_df.rename(columns=new_col_names, inplace=True)

    print("\tDataframe current shape = {}".format(sliced_df.shape))

    # d. Normalizing data
    print("6.d. Normalizing data: Converting to percetages")

    for col in sliced_df.columns:
        # Normalizing all but 'Target = Y', and Timestamp columns
        if (col != "Target" and col != 'Timestamp'):
            # normalizes the column according to its % of change (pct_change)
            sliced_df[col] = sliced_df[col].pct_change()
            sliced_df.dropna(inplace=True)

    # e. Scaling data
    print("6.e. Scaling data: Converting to a range from 0 to 1")
    for col in sliced_df.columns:
        # Scaling the values from 0 to 1
        if (col != "Target" and col != 'Timestamp'):
            sliced_df[col] = preprocessing.scale(sliced_df[col].values)
    sliced_df.dropna(inplace=True)

    # Restoring the values of Timestamps
    """
    """
    dates = pd.to_datetime(['2019-01-15 13:30:00'])
    (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    # Int64Index([1547559000], dtype='int64')
    """
    """
    t = pd.Timestamp("1970-01-01")
    sliced_df['Timestamp'] = (sliced_df['Timestamp'] - t) / pd.Timedelta('1s')

    return sliced_df
    """


def plotting_df(a_dataframe):
    """
    [Function that prints X and Y data from a Pandas DF]

    Args:
        a_dataframe ([pandas df]): [a valid pandas dataframe]

    Returns:
        Nothing
    """

    """
    import matplotlib.pyplot as plt

    print("5. Plotting data")
    x_data = pd.to_datetime(a_dataframe['Timestamp'], unit='s')

    try:
        y_data = a_dataframe['Close_USD']
    except BaseException:
        y_data = a_dataframe['Close']

    # 2. Form
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))

    # 3. font
    plt.rcParams.update({'font.size': 12})

    # 4.Labels
    plt.xlabel('Date')
    plt.ylabel('Closing value (US$)')
    plt.title("Bitcoin (BTC) - ฿")

    # 5. Display
    line_color = '#aec7e8'  # line color red
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(x_data, y_data, line_color, linewidth=1.45)
    plt.grid(True, ls='--', lw=.5, c='k', alpha=.13)
    plt.show()
    """


def saving_csv(df, new_name):
    """
    Function that saves a DF into a file of name 'new_name']

    Args:
        df ([pandas df]): [a valid pandas dataframe]
        new_name ([str]): [name of the file]

    Returns:
        None
    """

    """
    # df.reset_index(inplace=True, drop=True)
    new_file = "./" + new_name
    try:
        df.to_csv(new_file, index=False)
        MB = os.path.getsize(new_name) / 1000000
        print("7. Saving file: '{:>17}'\tSize = {} MB".format(new_name, MB))

    except BaseException as e:
        print(e)

    return None
    """
