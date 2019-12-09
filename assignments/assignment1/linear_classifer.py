import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops

    if predictions.ndim > 1:
        predictions -= np.max(predictions, axis=1).transpose().reshape((predictions.shape[0], 1))
        result = np.exp(predictions) / np.sum(np.exp(predictions), axis=1).transpose().reshape((predictions.shape[0], 1))
    else:
        predictions -= np.max(predictions)
        result = np.exp(predictions) / np.sum(np.exp(predictions))

    return result


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops

    if hasattr(target_index, '__len__'):
        ans = -np.sum(np.log(probs[np.arange(len(target_index)), target_index.reshape(1, -1)]))
    else:
        ans = -np.log(probs[target_index])

    return ans


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''

    loss = cross_entropy_loss(softmax(predictions.copy()), target_index)
    mask = np.zeros(predictions.shape)
    if hasattr(target_index, '__len__'):
        mask[np.arange(len(target_index)), target_index.reshape(1, -1)] = 1
    else:
        mask[target_index] = 1

    dprediction = softmax(predictions.copy()) - mask

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")

    loss = reg_strength * np.sum(W * W)
    grad = reg_strength * 2 * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")

    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = X.transpose().dot(dprediction)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):

            loss = 0

            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch_index in batches_indices:
                batch_X = X[batch_index]
                batch_y = y[batch_index]

                loss_softmax, dW_softmax = linear_softmax(batch_X, self.W, batch_y)
                loss_relur, dW_regul = l2_regularization(self.W, reg_strength=reg)
                loss += (loss_softmax + loss_relur)

                loss_history.append(loss)
                self.W -= learning_rate * (dW_softmax + dW_regul)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = X.dot(self.W).argmax(axis=1)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")

        return y_pred


if __name__ == '__main__':
    # TODO Implement linear_softmax function that uses softmax with cross-entropy for linear classifier
    # TODO: Implement LinearSoftmaxClassifier.fit function

    from assignments.assignment1.dataset import load_svhn, random_split_train_val

    def prepare_for_linear_classifier(train_X, test_X):
        train_flat = train_X.reshape(train_X.shape[0], -1).astype(np.float) / 255.0
        test_flat = test_X.reshape(test_X.shape[0], -1).astype(np.float) / 255.0

        # Subtract mean
        mean_image = np.mean(train_flat, axis=0)
        train_flat -= mean_image
        test_flat -= mean_image

        # Add another channel with ones as a bias term
        train_flat_with_ones = np.hstack([train_flat, np.ones((train_X.shape[0], 1))])
        test_flat_with_ones = np.hstack([test_flat, np.ones((test_X.shape[0], 1))])
        return train_flat_with_ones, test_flat_with_ones


    train_X, train_y, test_X, test_y = load_svhn("data", max_train=10000, max_test=1000)
    train_X, test_X = prepare_for_linear_classifier(train_X, test_X)
    # Split train into train and val
    train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val=1000)

    classifier = LinearSoftmaxClassifier()
    loss_history = classifier.fit(train_X, train_y, epochs=10, learning_rate=1e-3, batch_size=300, reg=1e1)
