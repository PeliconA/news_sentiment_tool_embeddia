"""
Author: Andraž Pelicon

This file serves as an example of how to use the trained news sentiment model on your data. First, you need
to install all the necessary dependencies according to the instructions in the README file of this repository.

For this example, we assume we have our data that need to be classified in a .tsv file in the 'data' folder.

"""

# We first import the function 'predict' from 'src' module."""
from src.predict import predict

import pandas as pd

def run_example():
    # We then load our data from the tsv file into a list, where each list element is an instance to be classified
    data = pd.read_csv("./data/example_news.tsv", sep="\t")
    instances_to_be_classified = data['data']

    # We then call the predict function sending our data as parameter. The function returns predictions for each
    # instance.
    predictions = predict(instances_to_be_classified)
    print(predictions)


if __name__ == "__main__":
    run_example()
