import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def data_cleaner(input_column_name, target_column_name, dataset):
    """
    Cleans the specified text column in a pandas DataFrame using a series of text preprocessing steps.

    Parameters:
    - input_column_name (str): The name of the column in the DataFrame to clean.
    - target_column_name (str): The name of the target column, which is checked for existence but not modified.
    - dataset (pd.DataFrame): The DataFrame containing the data to be cleaned.

    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    
    # Ensure that the dataset is a pandas DataFrame
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("The dataset should be a pandas DataFrame.")
    
    # Load NLTK stopwords
    stopwords_list = set(stopwords.words('english'))

    # Define a list of negation words to be preserved
    negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', 
                      'don\'t', 'doesn\'t', 'didn\'t', 'can\'t', 'won\'t', 'wouldn\'t', 'shouldn\'t', 'mustn\'t'}

    # Convert text to lowercase
    def to_lower_case(examples):
        return examples.str.lower()

    # Remove stop words from the text, preserving negation words
    def remove_stop_words(data, stopwords_list, negation_words):
        return data.apply(lambda text: ' '.join(word for word in text.split() if word.lower() not in stopwords_list or word.lower() in negation_words))

    # Remove special characters from the text
    def remove_special_characters(examples):
        return examples.apply(lambda text: re.sub(r'[^\w\s]', '', text))

    # Remove numbers from the text
    def remove_numbers(examples):
        return examples.apply(lambda text: re.sub(r'\d+', '', text))

    # Remove emojis from the text
    def remove_emojis(examples):
        return examples.apply(lambda text: text.encode('ascii', 'ignore').decode('ascii'))

    # Remove single characters from the text
    def remove_single_characters(examples):
        return examples.apply(lambda text: re.sub(r'\b\w\b', '', text).strip())

    # Remove repeated letters (e.g., "loooove" -> "love")
    def remove_repeated_letters(examples):
        return examples.apply(lambda text: re.sub(r'([a-zA-Z])\1{2,}', r'\1', text))

    # Remove extra spaces from the text
    def remove_extra_spaces(examples):
        return examples.apply(lambda text: ' '.join(text.split()))

    # Remove rare words that occur fewer than a specified number of times
    def remove_rare_words(examples, min_freq=10):
        all_words = [word for text in examples for word in text.split()]
        word_counts = Counter(all_words)
        return examples.apply(lambda text: ' '.join(word for word in text.split() if word_counts[word] >= min_freq))

    # Validate that the specified columns exist in the dataset
    if input_column_name not in dataset.columns or target_column_name not in dataset.columns:
        raise ValueError(f"Column '{input_column_name}' or '{target_column_name}' not found in dataset.")
    
    # Apply each text cleaning function in sequence
    dataset[input_column_name] = to_lower_case(dataset[input_column_name])
    dataset[input_column_name] = remove_special_characters(dataset[input_column_name])
    dataset[input_column_name] = remove_numbers(dataset[input_column_name])
    dataset[input_column_name] = remove_emojis(dataset[input_column_name])
    dataset[input_column_name] = remove_stop_words(dataset[input_column_name], stopwords_list, negation_words)
    dataset[input_column_name] = remove_rare_words(dataset[input_column_name], min_freq=10)
    dataset[input_column_name] = remove_extra_spaces(dataset[input_column_name])
    dataset[input_column_name] = remove_single_characters(dataset[input_column_name])
    dataset[input_column_name] = remove_repeated_letters(dataset[input_column_name])
    
    # Remove rows where the cleaned text column is empty
    dataset = dataset[dataset[input_column_name].str.strip() != ''].reset_index(drop=True)

    # Print a success message and a preview of the cleaned dataset
    print("=" * 50)
    print("Step 2: The data was cleaned successfully :)")
    print("=" * 50)
    print(dataset.head(5))

    return dataset