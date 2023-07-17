import numpy as np

def prepare_data_from_torchtext(data_source):
    """Takes torchtext data source and returns X and y data for both train and test set.
    
    Example usage:

    from torchtext.datasets import AG_NEWS 
    train_X, train_y, test_X, test_y = prepare_data_from_torchtext(AG_NEWS)
    """
    train, test = data_source("data")

    def to_X_y(data):
        y, X = zip(*data)  # The structure is (label, text) for some reason
        return np.array(X).reshape(-1, 1), np.array(y)
    
    return *to_X_y(train), *to_X_y(test)