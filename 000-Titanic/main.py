#!/usr/bin/env python3

import sys

# import pandas as pd

clean_sc = __import__('preprocess_data').clean_screen
read_arg = __import__('preprocess_data').read_name
open_zip = __import__('preprocess_data').extract_fromzip
save_lst = __import__('preprocess_data').save_csv_list
rec_name = __import__('preprocess_data').recover_name

read_csv = __import__('preprocess_data').read_csv
plotting = __import__('preprocess_data').plotting_df
preproc_ = __import__('preprocess_data').preprocess

data_seq = __import__('forecast_btc').data_seq
balances = __import__('forecast_btc').verify_data_balance
RNN_make = __import__('forecast_btc').build_RNN
train_md = __import__('forecast_btc').train_model

# pd.set_option('mode.chained_assignment', None)  # Avoid warnings


# 0. Clean screen
clean_sc()

# 1. Read the arguments for traininig
FILE_PATH = read_arg()
if (FILE_PATH == None):
    sys.exit(0)

# 2. Decompress ZIP file
CSV_LST = open_zip(FILE_PATH)
if (CSV_LST == None):
    sys.exit(0)

# 3. Serializing list of decompressed files
save_lst(CSV_LST, 'csv_files')

# 4. Recover the names of the CSV_files
print("3. Recovering files")
train_file = rec_name("train", CSV_LST)
test__file = rec_name("test", CSV_LST)

# 5. Read train CSV_file
train_df = read_csv("train")

"""
# 5. Plotting
plotting(df)
print()

# 6. Preprocessing the selected Dataframe (Slicing data)
main_df = preproc_(df, "Main DF")
print()
vali_df = preproc_(validation_df, "Validation DF")
print()

print("Bitcoin (฿) forecasting")
print("Part 2. Predicting values")
print("---------------------")
print()

# 7. Generate data sequences
train_X, train_Y = data_seq(main_df, WPH, "Training")
print()
valid_X, valid_Y = data_seq(vali_df, WPH, "Validation")
print()

# 8. Confirming sizes
balances(train_X, train_Y, valid_X, valid_Y)
print()

# 9. Building and compile the model
model = RNN_make(train_X)
print()

# 10. Train the model
train_md(model, train_X, train_Y, valid_X, valid_Y, FPH, WPH, E, B)
print()

# 11 Plotting new results

# 12. Saving data
# saving_csv(main_df, "main_DF.csv")
# saving_csv(vali_df, "validation_DF.csv")
"""
