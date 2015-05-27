import numpy as np
import theano
import ctc_cost
import theano.tensor as T
from numpy import testing
from itertools import izip, islice


floatX = theano.config.floatX


def test_log_add():
    x = T.scalar()
    y = T.scalar()
    z = ctc_cost._log_add(x, y)
    X = -3.0
    Y = -np.inf
    value = z.eval({x: X, y: Y})
    assert value == -3.0


def test_log_dot_matrix():
    x = T.matrix()
    y = T.matrix()
    z = ctc_cost._log_dot_matrix(y, x)
    X = np.asarray(np.random.normal(0, 1, (5, 4)), dtype=floatX)
    Y = np.asarray(np.random.normal(0, 1, (3, 5)), dtype=floatX)
    #Y = np.ones((3, 5), dtype=floatX) * 3
    value = z.eval({x: X, y: Y})
    np_value = np.log(np.dot(np.exp(Y), np.exp(X)))
    assert np.mean((value - np_value)**2) < 1e5


def test_log_dot_matrix_zeros():
    x = T.matrix()
    y = T.matrix()
    z = ctc_cost._log_dot_matrix(y, x)
    X = np.log(np.asarray(np.eye(5), dtype=floatX))
    Y = np.asarray(np.random.normal(0, 1, (3, 5)), dtype=floatX)
    #Y = np.ones((3, 5), dtype=floatX) * 3
    value = z.eval({x: X, y: Y})
    np_value = np.log(np.dot(np.exp(Y), np.exp(X)))
    assert np.mean((value - np_value)**2) < 1e5


def test_ctc_add_blanks():
    BATCHES = 3
    N_LABELS = 3
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    blanked_y, blanked_y_mask = ctc_cost._add_blanks(
        y=y,
        blank_symbol=1,
        y_mask=y_mask)
    Y = np.zeros((N_LABELS, BATCHES), dtype='int64')
    Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    Y_mask[-1, 0] = 0
    Blanked_y_mask = blanked_y_mask.eval({y_mask: Y_mask})
    Blanked_y = blanked_y.eval({y: Y})
    assert (Blanked_y == np.array([[1, 1, 1],
                                   [0, 0, 0],
                                   [1, 1, 1],
                                   [0, 0, 0],
                                   [1, 1, 1],
                                   [0, 0, 0],
                                   [1, 1, 1]], dtype='int32')).all()
    assert (Blanked_y_mask == np.array([[1., 1., 1.],
                                        [1., 1., 1.],
                                        [1., 1., 1.],
                                        [1., 1., 1.],
                                        [1., 1., 1.],
                                        [0., 1., 1.],
                                        [0., 1., 1.]], dtype=floatX)).all()


def test_ctc_symmetry_logscale():
    LENGTH = 5000
    BATCHES = 3
    CLASSES = 4
    y_hat = T.tensor3('features')
    input_mask = T.matrix('features_mask')
    y_hat_mask = input_mask
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    ctc_cost_t = ctc_cost.cost(y, y_hat, y_mask, y_hat_mask)

    Y_hat = np.zeros((LENGTH, BATCHES, CLASSES), dtype=floatX)
    Y_hat[:, :, 0] = .3
    Y_hat[:, :, 1] = .2
    Y_hat[:, :, 2] = .4
    Y_hat[:, :, 3] = .1
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    # default blank symbol is the highest class index (3 in this case)
    Y = np.repeat(np.array([0, 1, 2, 1, 2, 0, 2, 2, 2]),
                  BATCHES).reshape((9, BATCHES))
    # the masks for this test should be all ones.
    Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    forward_cost = ctc_cost_t.eval({y_hat: Y_hat, y: Y,
                                  y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    backward_cost = ctc_cost_t.eval({y_hat: Y_hat, y: Y[::-1],
                                   y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    testing.assert_almost_equal(forward_cost[0], backward_cost[0])
    assert not np.isnan(forward_cost[0])
    assert not np.isnan(backward_cost[0])
    assert not np.isinf(np.abs(forward_cost[0]))
    assert not np.isinf(np.abs(backward_cost[0]))


def test_ctc_symmetry():
    LENGTH = 20
    BATCHES = 3
    CLASSES = 4
    y_hat = T.tensor3('features')
    input_mask = T.matrix('features_mask')
    y_hat_mask = input_mask
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    ctc_cost_t = ctc_cost.cost(y, y_hat, y_mask, y_hat_mask, log_scale=False)

    Y_hat = np.zeros((LENGTH, BATCHES, CLASSES), dtype=floatX)
    Y_hat[:, :, 0] = .3
    Y_hat[:, :, 1] = .2
    Y_hat[:, :, 2] = .4
    Y_hat[:, :, 3] = .1
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    # default blank symbol is the highest class index (3 in this case)
    Y = np.repeat(np.array([0, 1, 2, 1, 2, 0, 2, 2, 2]),
                  BATCHES).reshape((9, BATCHES))
    # the masks for this test should be all ones.
    Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    forward_cost = ctc_cost_t.eval({y_hat: Y_hat, y: Y,
                                  y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    backward_cost = ctc_cost_t.eval({y_hat: Y_hat, y: Y[::-1],
                                   y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    testing.assert_almost_equal(forward_cost[0], backward_cost[0])
    assert not np.isnan(forward_cost[0])
    assert not np.isnan(backward_cost[0])
    assert not np.isinf(np.abs(forward_cost[0]))
    assert not np.isinf(np.abs(backward_cost[0]))


def test_ctc_exact_log_scale():
    LENGTH = 4
    BATCHES = 1
    CLASSES = 2
    y_hat = T.tensor3('features')
    input_mask = T.matrix('features_mask')
    y_hat_mask = input_mask
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    ctc_cost_t = ctc_cost.cost(y, y_hat, y_mask, y_hat_mask, log_scale=True)

    Y_hat = np.zeros((LENGTH, BATCHES, CLASSES), dtype=floatX)
    Y_hat[:, :, 0] = .7
    Y_hat[:, :, 1] = .3
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    # default blank symbol is the highest class index (3 in this case)
    Y = np.zeros((2, 1), dtype='int64')
    # -0-0
    # 0-0-
    # 0--0
    # 0-00
    # 00-0
    answer = np.log(3 * (.3 * .7)**2 + 2 * .3 * .7**3)
    Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    forward_cost = ctc_cost_t.eval({y_hat: Y_hat, y: Y,
                                  y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    backward_cost = ctc_cost_t.eval({y_hat: Y_hat, y: Y[::-1],
                                   y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    assert not np.isnan(forward_cost[0])
    assert not np.isnan(backward_cost[0])
    assert not np.isinf(np.abs(forward_cost[0]))
    assert not np.isinf(np.abs(backward_cost[0]))
    testing.assert_almost_equal(-forward_cost[0], answer)
    testing.assert_almost_equal(-backward_cost[0], answer)


def test_ctc_exact():
    LENGTH = 4
    BATCHES = 1
    CLASSES = 2
    y_hat = T.tensor3('features')
    input_mask = T.matrix('features_mask')
    y_hat_mask = input_mask
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    ctc_cost_t = ctc_cost.cost(y, y_hat, y_mask, y_hat_mask, log_scale=False)

    Y_hat = np.zeros((LENGTH, BATCHES, CLASSES), dtype=floatX)
    Y_hat[:, :, 0] = .7
    Y_hat[:, :, 1] = .3
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    # default blank symbol is the highest class index (3 in this case)
    Y = np.zeros((2, 1), dtype='int64')
    # -0-0
    # 0-0-
    # 0--0
    # 0-00
    # 00-0
    answer = np.log(3 * (.3 * .7)**2 + 2 * .3 * .7**3)
    Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    forward_cost = ctc_cost_t.eval({y_hat: Y_hat, y: Y,
                                  y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    backward_cost = ctc_cost_t.eval({y_hat: Y_hat, y: Y[::-1],
                                   y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    assert not np.isnan(forward_cost[0])
    assert not np.isnan(backward_cost[0])
    assert not np.isinf(np.abs(forward_cost[0]))
    assert not np.isinf(np.abs(backward_cost[0]))
    testing.assert_almost_equal(-forward_cost[0], answer)
    testing.assert_almost_equal(-backward_cost[0], answer)


def test_ctc_log_path_probabs():
    LENGTH = 10
    BATCHES = 3
    CLASSES = 2
    N_LABELS = 3
    y_hat = T.tensor3('features')
    input_mask = T.matrix('features_mask')
    y_hat_mask = input_mask
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    blanked_y, blanked_y_mask = ctc_cost._add_blanks(
        y=y,
        blank_symbol=1,
        y_mask=y_mask)
    p = ctc_cost._log_path_probabs(blanked_y, y_hat, blanked_y_mask, y_hat_mask, 1)
    Y_hat = np.zeros((LENGTH, BATCHES, CLASSES + 1), dtype=floatX)
    Y_hat[:, :, 0] = .7
    Y_hat[:, :, 1] = .2
    Y_hat[:, :, 2] = .1
    Y = np.zeros((N_LABELS, BATCHES), dtype='int64')
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    Y_hat_mask[-2:, 0] = 0
    Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    forward_probs = p.eval({y_hat: Y_hat, y: Y,
                            y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    assert forward_probs[-2, 0, 0] == -np.inf
    Y_mask[-1] = 0
    forward_probs_y_mask = p.eval({y_hat: Y_hat, y: Y,
                                   y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    assert forward_probs_y_mask[-1, 1, -2] == -np.inf
    assert not np.isnan(forward_probs).any()


def test_ctc_log_forward_backward():
    LENGTH = 8
    BATCHES = 4
    CLASSES = 2
    N_LABELS = 3
    y_hat = T.tensor3('features')
    input_mask = T.matrix('features_mask')
    y_hat_mask = input_mask
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    blanked_y, blanked_y_mask = ctc_cost._add_blanks(
        y=y,
        blank_symbol=1,
        y_mask=y_mask)
    f, b = ctc_cost._log_forward_backward(blanked_y, y_hat,
                                          blanked_y_mask, y_hat_mask, CLASSES)
    Y_hat = np.zeros((LENGTH, BATCHES, CLASSES + 1), dtype=floatX)
    Y_hat[:, :, 0] = .7
    Y_hat[:, :, 1] = .2
    Y_hat[:, :, 2] = .1
    Y_hat[3, :, 0] = .3
    Y_hat[3, :, 1] = .4
    Y_hat[3, :, 2] = .3
    Y = np.zeros((N_LABELS, BATCHES), dtype='int64')
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    Y_hat_mask[-2:] = 0
    Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    Y_mask[-2:, 0] = 0
    y_prob = ctc_cost._class_batch_to_labeling_batch(blanked_y,
                                                    y_hat,
                                                    y_hat_mask)
    forward_probs = f.eval({y_hat: Y_hat, y: Y,
                            y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    backward_probs = b.eval({y_hat: Y_hat, y: Y,
                            y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    y_probs = y_prob.eval({y_hat: Y_hat, y: Y, y_hat_mask: Y_hat_mask})
    assert not ((forward_probs + backward_probs)[:, 0, :] == -np.inf).all()
    marg = forward_probs + backward_probs - np.log(y_probs)
    forward_probs = np.exp(forward_probs)
    backward_probs = np.exp(backward_probs)
    L = (forward_probs * backward_probs[::-1][:, :, ::-1] / y_probs).sum(2)
    assert not np.isnan(forward_probs).any()


def finite_diff(Y, Y_hat, Y_mask, Y_hat_mask, eps=1e-2, n_steps=None):
    y_hat = T.tensor3('features')
    y_hat_mask = T.matrix('features_mask')
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    ctc_cost_t = ctc_cost.cost(y, y_hat, y_mask, y_hat_mask)
    get_cost = theano.function([y, y_hat, y_mask, y_hat_mask],
                               ctc_cost_t.sum())
    diff_grad = np.zeros_like(Y_hat)
    
    for grad, val in islice(izip(np.nditer(diff_grad, op_flags=['readwrite']),
                                 np.nditer(Y_hat, op_flags=['readwrite'])),
                            0, n_steps):
        val += eps
        error_inc = get_cost(Y, Y_hat, Y_mask, Y_hat_mask)
        val -= 2.0 * eps
        error_dec = get_cost(Y, Y_hat, Y_mask, Y_hat_mask)
        grad[...] = .5 * (error_inc - error_dec) / eps
        val += eps

    return diff_grad


def test_ctc_class_batch_to_labeling_batch():
    LENGTH = 20
    BATCHES = 4
    CLASSES = 2
    LABELS = 2
    y_hat = T.tensor3()
    y_hat_mask = T.matrix('features_mask')
    y = T.lmatrix('phonemes')
    y_labeling = ctc_cost._class_batch_to_labeling_batch(y, y_hat, y_hat_mask)
    Y_hat = np.zeros((LENGTH, BATCHES, CLASSES + 1), dtype=floatX)
    Y = np.zeros((2, BATCHES), dtype='int64')
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    Y_hat_mask[-5:] = 0
    Y_labeling = y_labeling.eval({y_hat: Y_hat, y: Y, y_hat_mask: Y_hat_mask})
    assert Y_labeling.shape == (LENGTH, BATCHES, LABELS)


def test_ctc_labeling_batch_to_class_batch():
    LENGTH = 20
    BATCHES = 4
    CLASSES = 2
    LABELS = 2
    y_labeling = T.tensor3()
    y = T.lmatrix('phonemes')
    y_hat = ctc_cost._labeling_batch_to_class_batch(y, y_labeling, CLASSES + 1)
    Y_labeling = np.zeros((LENGTH, BATCHES, LABELS), dtype=floatX)
    Y = np.zeros((2, BATCHES), dtype='int64')
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    Y_hat_mask[-5:] = 0
    Y_hat = y_hat.eval({y_labeling: Y_labeling, y: Y})
    assert Y_hat.shape == (LENGTH, BATCHES, CLASSES + 1)


def test_ctc_targets():
    LENGTH = 20
    BATCHES = 4
    CLASSES = 2
    y_hat = T.tensor3('features')
    input_mask = T.matrix('features_mask')
    y_hat_mask = input_mask
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    ctc_target = ctc_cost.get_targets(y, T.log(y_hat), y_mask, y_hat_mask)
    Y_hat = np.zeros((LENGTH, BATCHES, CLASSES + 1), dtype=floatX)
    Y_hat[:, :, 0] = .7
    Y_hat[:, :, 1] = .2
    Y_hat[:, :, 2] = .1
    Y_hat[3, :, 0] = .3
    Y_hat[3, :, 1] = .4
    Y_hat[3, :, 2] = .3
    Y = np.zeros((2, BATCHES), dtype='int64')
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    Y_hat_mask[-5:] = 0
    # default blank symbol is the highest class index (3 in this case)
    Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    target = ctc_target.eval({y_hat: Y_hat, y: Y,
                              y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    # Note that this part is the same as the cross entropy gradient
    grad = -target / Y_hat
    test_grad = finite_diff(Y, Y_hat, Y_mask, Y_hat_mask, eps=1e-2, n_steps=5)
    testing.assert_almost_equal(grad.flatten()[:5],
                                test_grad.flatten()[:5], decimal=3)


def test_ctc_pseudo_cost():
    LENGTH = 500
    BATCHES = 40
    CLASSES = 2
    N_LABELS = 45
    y_hat = T.tensor3('features')
    input_mask = T.matrix('features_mask')
    y_hat_mask = input_mask
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    pseudo_cost = ctc_cost.pseudo_cost(y, y_hat, y_mask, y_hat_mask)

    Y_hat = np.zeros((LENGTH, BATCHES, CLASSES + 1), dtype=floatX)
    Y_hat[:, :, 0] = .75
    Y_hat[:, :, 1] = .2
    Y_hat[:, :, 2] = .05
    Y_hat[3, 0, 0] = .3
    Y_hat[3, 0, 1] = .4
    Y_hat[3, 0, 2] = .3
    Y = np.zeros((N_LABELS, BATCHES), dtype='int64')
    Y[25:, :] = 1
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    Y_hat_mask[-5:] = 0
    # default blank symbol is the highest class index (3 in this case)
    Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    Y_mask[30:] = 0
    cost = pseudo_cost.eval({y_hat: Y_hat, y: Y,
                             y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    pseudo_grad = T.grad(ctc_cost.pseudo_cost(y, y_hat,
                                              y_mask, y_hat_mask).sum(),
                         y_hat)
    #test_grad2 = pseudo_grad.eval({y_hat: Y_hat, y: Y,
    #                               y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    # TODO: write some more meaningful asserts here
    assert cost.sum() > 0


def test_ctc_pseudo_cost_skip_softmax_stability():
    LENGTH = 500
    BATCHES = 40
    CLASSES = 2
    N_LABELS = 45
    y_hat = T.tensor3('features')
    input_mask = T.matrix('features_mask')
    y_hat_mask = input_mask
    y = T.lmatrix('phonemes')
    y_mask = T.matrix('phonemes_mask')
    pseudo_cost = ctc_cost.pseudo_cost(y, y_hat, y_mask, y_hat_mask,
                                       skip_softmax=True)

    Y_hat = np.asarray(np.random.normal(0, 1, (LENGTH, BATCHES, CLASSES + 1)),
                       dtype=floatX)
    Y = np.zeros((N_LABELS, BATCHES), dtype='int64')
    Y[25:, :] = 1
    Y_hat_mask = np.ones((LENGTH, BATCHES), dtype=floatX)
    Y_hat_mask[-5:] = 0
    # default blank symbol is the highest class index (3 in this case)
    Y_mask = np.asarray(np.ones_like(Y), dtype=floatX)
    Y_mask[30:] = 0
    pseudo_grad = T.grad(pseudo_cost.sum(), y_hat)
    test_grad = pseudo_grad.eval({y_hat: Y_hat, y: Y,
                                  y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    y_hat_softmax = T.exp(y_hat) / T.exp(y_hat).sum(2)[:, :, None]
    pseudo_cost2 = ctc_cost.pseudo_cost(y, y_hat_softmax, y_mask, y_hat_mask,
                                        skip_softmax=False)
    pseudo_grad2 = T.grad(pseudo_cost2.sum(), y_hat)
    test_grad2 = pseudo_grad2.eval({y_hat: Y_hat, y: Y,
                                    y_hat_mask: Y_hat_mask, y_mask: Y_mask})
    testing.assert_almost_equal(test_grad, test_grad2, decimal=4)
