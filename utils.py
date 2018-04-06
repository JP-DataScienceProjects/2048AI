import numpy as np

class Utils():
    @staticmethod
    def one_hot(data, classes):
        encoded = np.zeros((len(data), classes))
        encoded[np.arange(encoded.shape[0]), np.array(data)] = 1
        return encoded

    @staticmethod
    def softmax(probabilities):
        e_probs = np.exp(probabilities)
        return e_probs / np.sum(e_probs)