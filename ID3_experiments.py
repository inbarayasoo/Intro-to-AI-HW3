from ID3 import ID3
from utils import *

"""
Make the imports of python packages needed
"""

"""
========================================================================
========================================================================
                              Experiments 
========================================================================
========================================================================
"""
target_attribute = 'diagnosis'


# ========================================================================
def basic_experiment(x_train, y_train, x_test, y_test, formatted_print=False):
    """
    Use ID3 model, to train on the training dataset and evaluating the accuracy in the test set.
    """

    # TODO:
    #  - Instate ID3 decision tree instance.
    #  - Fit the tree on the training data set.
    #  - Test the model on the test set (evaluate the accuracy) and print the result.

    acc = None

    # ====== YOUR CODE: ======
    id3 = ID3(attributes_names)
    id3.fit(x_train, y_train)

    y_pred = id3.predict(x_test)
    acc = accuracy(y_test, y_pred)
    # ========================

    assert acc > 0.9, 'you should get an accuracy of at least 90% for the full ID3 decision tree'
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)


# ========================================================================
if __name__ == '__main__':
    attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    data_split = get_dataset_split(train_dataset, test_dataset, target_attribute)

    """
    Usages helper:
    (*) To get the results in “informal” or nicely printable string representation of an object
        modify the call "utils.set_formatted_values(value=False)" from False to True and run it
    """
    formatted_print = True
    basic_experiment(*data_split, formatted_print)
