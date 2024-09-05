import pandas as pd

def data_importer(path):
    # Read the CSV file
    dataset = pd.read_csv(path)

    # Print success message and dataset preview
    print("=" * 50)
    print("Step 1: The data loaded successfully :)")
    print("=" * 50)
    print(dataset.head(5))  # Display the first 5 rows of the dataset

    return dataset