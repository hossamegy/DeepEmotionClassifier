import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def data_preprocessor(input_column_name, target_column_name, dataset, num_words, oov_token, type_padding, type_truncating, maxlen):
    """
    Preprocesses text data by tokenizing, padding sequences, and encoding labels.

    Args:
        input_column_name (str): The column name in the dataset containing the input text data.
        target_column_name (str): The column name in the dataset containing the target labels.
        dataset (pd.DataFrame): The dataset containing the input and target data.
        num_words (int): The maximum number of words to keep in the tokenizer.
        oov_token (str): The token used for out-of-vocabulary words.
        type_padding (str): The type of padding ('pre' or 'post').
        type_truncating (str): The type of truncating ('pre' or 'post').
        maxlen (int): The maximum length of sequences after padding/truncating.

    Returns:
        tuple: Contains the preprocessed input sequences, one-hot encoded labels, label encoder, and tokenizer.
    """
    # Ensure the text data is in string format
    dataset[input_column_name] = dataset[input_column_name].astype(str)

    # Initialize and fit the tokenizer
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(dataset[input_column_name])

    # Split the data into training, validation, and test sets
    x_train, x_validation, y_train, y_validation = train_test_split(
        dataset[input_column_name],
        dataset[target_column_name],
        test_size=0.40,
        random_state=42
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_validation.tolist(),
        y_validation.tolist(),
        test_size=0.50,
        random_state=42
    )

    # Convert text sequences to padded sequences
    all_input_sequence_data = {
        'train': pad_sequences(
            tokenizer.texts_to_sequences(x_train),
            padding=type_padding,
            truncating=type_truncating,
            maxlen=maxlen
        ),
        'val': pad_sequences(
            tokenizer.texts_to_sequences(x_val),
            padding=type_padding,
            truncating=type_truncating,
            maxlen=maxlen
        ),
        'test': pad_sequences(
            tokenizer.texts_to_sequences(x_test),
            padding=type_padding,
            truncating=type_truncating,
            maxlen=maxlen
        )
    }

    # Initialize and fit the label encoder
    label_encoder = LabelEncoder()
    all_labels = dataset[target_column_name].astype(str).tolist()
    label_encoder.fit(all_labels)

    # Transform labels to encoded format
    y_train_encoded = label_encoder.transform(np.array(y_train, dtype=str))
    y_val_encoded = label_encoder.transform(np.array(y_val, dtype=str))
    y_test_encoded = label_encoder.transform(np.array(y_test, dtype=str))

    num_classes = len(label_encoder.classes_)

    # Convert encoded labels to one-hot format
    all_label = {
        'train': to_categorical(y_train_encoded, num_classes=num_classes),
        'val': to_categorical(y_val_encoded, num_classes=num_classes),
        'test': to_categorical(y_test_encoded, num_classes=num_classes)
    }

    # Print success message and details
    print("=" * 52)
    print("Step 3: Data preprocessing was done successfully :)")
    print("=" * 52)
    print("Label classes:", label_encoder.classes_)
    print("Input sequence data keys:", all_input_sequence_data.keys())
    
    # Save tokenizer and label encoder
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)

    with open('label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)

    return all_input_sequence_data, all_label, label_encoder, tokenizer
