

import pandas as pd


def load_smr_triggers(file_name):

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)

    if 'imgs' in file_name:
        print("Substring 'imgs' exists in file_name")
        # rename columns
        new_column_names = {'t0': 'TrialStart', 's0': 'StimOn', 's2': 'StimOff', 't1': 'TrialEnd', 'iR': 'TrialSuccess'}
        df = df.rename(columns=new_column_names)

    else:
        print("Substring 'imgs' does not exist in file_name")
        # rename columns
        new_column_names = {'v': 'IV', 't0': 'TrialStart','s1': 'TrialSuccess', 's0': 'StimOn', 's2': 'StimOff', 't1': 'TrialEnd'}
        df = df.rename(columns=new_column_names)

    return df

