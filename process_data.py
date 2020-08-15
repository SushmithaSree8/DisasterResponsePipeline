import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	"""
	INPUT: Datasets (csv) to be read
	OUTPUT: Merged dataframe of the two input datasets
	"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):

	"""
	INPUT: Dataframe to be cleaned
	OUTPUT: CLeaned Dataframe

	The function cleans the category column of dataframe so each category is in separate column
	and drops the duplicate data
	"""
	
    #creating a dataframe categories with each category as a column
    categories = df.categories.str.split(pat=';', expand=True)

    #extracting a list of new column names for categories
    row = categories.iloc[0,:]
    #renaming columns of categories
    categories.columns = row.apply(lambda x: x.split('-')[0])

    # set each value 1 or 0 based on the last character of the string
    for column in categories:
    	categories[column] = categories[column].apply(lambda x: x.split('-')[1] )
    	categories[column] = pd.to_numeric(categories[column])

    #replacing category column in df with the separated columns in categories
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df,categories], axis=1)

    #dropping duplicates
    df.drop_duplicates(inplace=True)

    return df




def save_data(df, database_filename):
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()