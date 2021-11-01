import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' Loads the 2 dataframes in the args. Returns a joined dataframe.'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner')
    return df


def clean_data(df):
    ''' Takes dataframe as an arg, returns a clean dataframe.'''
    # Create a new dataframe that contains just the categories
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    # Taking first row in categories df, splitting each category name to make a column name out of it
    row = categories.iloc[0]
    column_names = []
    for i in row:
        column_names.append(i.split('-')[0])
    categories.columns = column_names
    # Changing the values in the columns to be either 0 or 1
    for i in categories:
        categories[i] = categories[i].str[-1]
        categories[i] = categories[i].astype(np.int)
    df.drop('categories', inplace=True, axis=1)
    df = pd.concat([df, categories], axis=1)
    print('Duplicates before cleaning: {}'.format(
        df.duplicated(subset='id', keep='first').sum()))
    df.drop_duplicates(subset='id', keep='first', inplace=True)
    print('Duplicates after cleaning: {}'.format(
        df.duplicated(subset='id', keep='first').sum()))

    #Related column has a problem with having a class other than 0 or 1
    print(df.related.unique())
        #We'll replace the 2 values with 1
    df['related'].replace({2: 1}, inplace=True)
    #Lets check for uniques again
    print(df.related.unique())
    return df


def save_data(df, database_filename):
    ''' Takes a dataframe and saves it into a database.'''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print(df.head)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
