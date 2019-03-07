import pandas as pd
import numpy as np
from helpers import save_dataset
from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.3, seed=42):
    """Prepares a data frame for model training and testing by converting data
    to Numpy arrays and splitting into train and test sets.

    Args:
        df (pandas.DataFrame): Source data frame.
        test_size (float): Size of test set as a percentage of total samples.
        seed (int): Seed for random state in split.
    Returns:
        X_train (numpy.Array): Training features.
        X_test (numpy.Array): Test features.
        y_train (numpy.Array): Training labels.
        y_test (numpy.Array): Test labels.

    """
    # convert data frame to Numpy array and split X and y
    X_data = df.drop(columns='class').values
    y_data = df['class'].values

    # split into train and test sets, ensuring that composition of classes in
    # original dataset is maintained in the splits
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=seed,
        stratify=y_data
    )

    return X_train, X_test, y_train, y_test


def map_values(value):
    if int(value) < 9:
        return 0
    if int(value) is 9 or int(value) is 10:
        return 1
    return 2


if __name__ == '__main__':
    print('Processing \n')
    tdir = 'data'

    names = ['abalone', 'contraceptive']
    for n in names:
        df = pd.read_csv(
            '../data/{}.csv'.format(n)
        )

        X_train, X_test, y_train, y_test = split_data(df)

        train_df = pd.DataFrame(
            np.append(X_train, y_train.reshape(-1, 1), 1)
        )
        test_df = pd.DataFrame(
            np.append(X_test, y_test.reshape(-1, 1), 1)
        )

        print('SHAPE OF test')
        print(test_df.shape)

        print('SHAPE OF Train')
        print(train_df.shape)

        save_dataset(train_df, '{}_train.csv'.format(n), sep=',',
                     subdir=tdir, header=False)
        save_dataset(test_df, '{}_test.csv'.format(n), sep=',',
                     subdir=tdir, header=False)
