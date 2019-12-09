import numpy as np


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        mask = np.zeros(x.shape)
        mask[ix] = delta
        x_plus_delta = x + mask
        x_minus_delta = x - mask
        f_plus_delta = f(x_plus_delta)[0]
        f_minus_delta = f(x_minus_delta)[0]
        numeric_grad_at_ix = (f_plus_delta - f_minus_delta) / (2 * delta)

        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True


if __name__ == '__main__':

    def square(x):
        return float(x * x), 2 * x

    print(check_gradient(square, np.array([3.0])))


    def array_sum(x):
        assert x.shape == (2,), x.shape
        return np.sum(x), np.ones_like(x)


    print(check_gradient(array_sum, np.array([3.0, 2.0])))


    def array_2d_sum(x):
        assert x.shape == (2, 2)
        return np.sum(x), np.ones_like(x)

    print(check_gradient(array_2d_sum, np.array([[3.0, 2.0], [1.0, 0.0]])))
