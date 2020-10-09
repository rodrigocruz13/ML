#!/usr/bin/env python3

import sys

# import pandas as pd

clean_sc = __import__('preprocess_data').clean_screen
maketree = __import__('preprocess_data').files_tree
read_arg = __import__('preprocess_data').read_args
open_zip = __import__('preprocess_data').open_zip
rec_name = __import__('preprocess_data').recover_name

read_csv = __import__('preprocess_data').read_csv
plotting = __import__('preprocess_data').plotting_df
preproc_ = __import__('preprocess_data').preprocess

#data_seq = __import__('forecast_btc').data_seq
#balances = __import__('forecast_btc').verify_data_balance
#RNN_make = __import__('forecast_btc').build_RNN
#train_md = __import__('forecast_btc').train_model

# pd.set_option('mode.chained_assignment', None)  # Avoid warnings


# 0. Clean screen
clean_sc()

# 1. Generate file structure
tree = maketree()

# 2. Read the arguments for traininig
args = read_arg(tree)
if (args == None):
    sys.exit(0)

# 3. Decompress ZIP file
paths = open_zip(args)
if (len(paths) == 0 or paths == None):
    sys.exit(0)

train_file, tests_file = rec_name("train", paths), rec_name("test", paths)
train_df, val_df = read_csv(train_file), read_csv(tests_file)

dfs, names  = [train_df, val_df], ["train", "validation"]
# 4. Preprocessing DFs: Correcting, Completing, Creating, and Converting
preprocessed_dfs = preproc_(dfs, names)
print()

"""
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
