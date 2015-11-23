"""
CTC-Connectionist Temporal Classification

Code provided by "Mohammad Pezeshki" and "Philemon Brakel"- May. 2015 -
Montreal Institute for Learning Algorithms

Referece: Graves, Alex, et al. "Connectionist temporal classification:
labelling unsegmented sequence data with recurrent neural networks."
Proceedings of the 23rd international conference on Machine learning.
ACM, 2006.

Credits: Shawn Tan, Rakesh Var

This code is distributed without any warranty, express or implied.
"""
"""Connectionist Temporal Classification
y_hat : T x B x C+1
y : L x B
y_hat_mask : T x B
y_mask : L x B
"""
import theano
import numpy
from theano import tensor
from theano import tensor as T


floatX = theano.config.floatX


def get_targets(y, log_y_hat, y_mask, y_hat_mask):
    """
    Returns the target values according to the CTC cost with respect to y_hat.
    Note that this is part of the gradient with respect to the softmax output
    and not with respect to the input of the original softmax function.
    All computations are done in log scale
    """
    num_classes = log_y_hat.shape[2] - 1
    blanked_y, blanked_y_mask = _add_blanks(
        y=y,
        blank_symbol=num_classes,
        y_mask=y_mask)

    log_alpha, log_beta = _log_forward_backward(blanked_y,
                                                log_y_hat, blanked_y_mask,
                                                y_hat_mask, num_classes)
    # explicitly not using a mask to prevent inf - inf
    y_prob = _class_batch_to_labeling_batch(blanked_y, log_y_hat,
                                            y_hat_mask=None)
    marginals = log_alpha + log_beta - y_prob
    max_marg = marginals.max(2)
    max_marg = T.switch(T.le(max_marg, -numpy.inf), 0, max_marg)
    log_Z = T.log(T.exp(marginals - max_marg[:, :, None]).sum(2))
    log_Z = log_Z + max_marg
    log_Z = T.switch(T.le(log_Z, -numpy.inf), 0, log_Z)
    targets = _labeling_batch_to_class_batch(blanked_y,
                                             T.exp(marginals -
                                                   log_Z[:, :, None]),
                                             num_classes + 1)
    return targets


def pseudo_cost(y, y_hat, y_mask, y_hat_mask, skip_softmax=False):
    """
    Training objective.
    Computes the marginal label probabilities and returns the
    cross entropy between this distribution and y_hat, ignoring the
    dependence of the two.
    This cost should have the same gradient but it should be more
    numerically stable.

    Here's how it works:

    Say delta_y is the gradient we want theano to return with respect to
    the input y and let's assume both variables are vectors. By simply
    computing dot(delta_y, y), we obtain a cost with gradient delta_y.

    Parameters
    ----------
    y : matrix (L, B)
        the target label sequences
    y_hat : tensor3 (T, B, C)
        class probabily distribution sequences, potentially in log domain
    y_mask : matrix (L, B)
        indicates which values of y to use
    y_hat_mask : matrix (T, B)
        indicates the lenghts of the sequences in y_hat
    skip_softmax : bool
        whether to interpret y_hat as probabilities or unnormalized energy
        values. The latter might be more numerically stable and efficient
        because it avoids the computation of the explicit cost and softmax
        gradients. You probably want to use this.
    """
    if skip_softmax:
        y_hat_softmax = (T.exp(y_hat - y_hat.max(2)[:, :, None]) /
                         T.exp(y_hat -
                               y_hat.max(2)[:, :, None]).sum(2)[:, :, None])
        y_hat_safe = y_hat - y_hat.max(2)[:, :, None]
        log_y_hat_softmax = (y_hat_safe -
                             T.log(T.exp(y_hat_safe).sum(2))[:, :, None])
        targets = get_targets(y, log_y_hat_softmax, y_mask, y_hat_mask)
    else:
        y_hat_softmax = y_hat
        targets = get_targets(y, (T.log(y_hat) -
                                  T.log(y_hat.sum(2)[:, :, None])),
                              y_mask, y_hat_mask)

    mask = y_hat_mask[:, :, None]
    if skip_softmax:
        y_hat_grad = y_hat_softmax - targets
        return (y_hat * mask *
                theano.gradient.disconnected_grad(y_hat_grad)).sum(0).sum(1)
    return -T.sum(theano.gradient.disconnected_grad(targets) *
                  T.log(y_hat**mask), axis=0).sum(1)


def sequence_log_likelihood(y, y_hat, y_mask, y_hat_mask, blank_symbol,
                            log_scale=True):
    """
    Based on code from Shawn Tan.
    Credits to Kyle Kastner as well.

    This function computes the CTC log likelihood for a sequence that has
    been augmented with blank labels.

    
    """
    y_hat_mask_len = tensor.sum(y_hat_mask, axis=0, dtype='int32')
    y_mask_len = tensor.sum(y_mask, axis=0, dtype='int32')

    if log_scale:
        log_probabs = _log_path_probabs(y, T.log(y_hat),
                                        y_mask, y_hat_mask,
                                        blank_symbol)
        batch_size = log_probabs.shape[1]

        # Add the probabilities of the final time steps to get the total
        # sequence likelihood.
        log_labels_probab = _log_add(
            log_probabs[y_hat_mask_len - 1,
                        tensor.arange(batch_size),
                        y_mask_len - 1],
            log_probabs[y_hat_mask_len - 1,
                        tensor.arange(batch_size),
                        y_mask_len - 2])
    else:
        probabilities = _path_probabs(y, y_hat,
                                      y_mask, y_hat_mask,
                                      blank_symbol)
        batch_size = probabilities.shape[1]
        labels_probab = (probabilities[y_hat_mask_len - 1,
                                       tensor.arange(batch_size),
                                       y_mask_len - 1] +
                         probabilities[y_hat_mask_len - 1,
                                       tensor.arange(batch_size),
                                       y_mask_len - 2])
        log_labels_probab = tensor.log(labels_probab)
    return log_labels_probab


def cost(y, y_hat, y_mask, y_hat_mask, log_scale=True):
    """
    Training objective.
    Computes the CTC cost using just the forward computations.
    The difference between this function and the vanilla 'cost' function
    is that this function adds blanks first.

    Note: don't try to compute the gradient of this version of the cost!

    ----
    Parameters
    ----------
    y : matrix (L, B)
        the target label sequences
    y_hat : tensor3 (T, B, C)
        class probabily distribution sequences
    y_mask : matrix (L, B)
        indicates which values of y to use
    y_hat_mask : matrix (T, B)
        indicates the lenghts of the sequences in y_hat
    log_scale : bool
        uses log domain computations if True
    
    """
    num_classes = y_hat.shape[2] - 1
    blanked_y, blanked_y_mask = _add_blanks(
        y=y,
        blank_symbol=num_classes,
        y_mask=y_mask)
    final_cost = -sequence_log_likelihood(blanked_y, y_hat,
                                          blanked_y_mask, y_hat_mask,
                                          num_classes,
                                          log_scale=log_scale)
    return final_cost


def _add_blanks(y, blank_symbol, y_mask=None):
    """Add blanks to a matrix and updates mask
    Input shape: L x B
    Output shape: 2L+1 x B
    """
    # for y
    y_extended = y.T.dimshuffle(0, 1, 'x')
    blanks = tensor.zeros_like(y_extended) + blank_symbol
    concat = tensor.concatenate([y_extended, blanks], axis=2)
    res = concat.reshape((concat.shape[0],
                          concat.shape[1] * concat.shape[2])).T
    begining_blanks = tensor.zeros((1, res.shape[1])) + blank_symbol
    blanked_y = tensor.concatenate([begining_blanks, res], axis=0)
    # for y_mask
    if y_mask is not None:
        y_mask_extended = y_mask.T.dimshuffle(0, 1, 'x')
        concat = tensor.concatenate([y_mask_extended,
                                     y_mask_extended], axis=2)
        res = concat.reshape((concat.shape[0],
                              concat.shape[1] * concat.shape[2])).T
        begining_blanks = tensor.ones((1, res.shape[1]), dtype=floatX)
        blanked_y_mask = tensor.concatenate([begining_blanks, res], axis=0)
    else:
        blanked_y_mask = None
    return blanked_y.astype('int32'), blanked_y_mask


def _class_batch_to_labeling_batch(y, y_hat, y_hat_mask=None):
    """
    Convert (T, B, C) tensor into (T, B, L) tensor.

    In other words, convert lattice of class probabilities into a lattice
    of label probabilities costrained by the sequence y.

    Notes
    -----
    T: number of time steps
    B: batch size
    L: length of label sequence
    C: number of classes
    Parameters
    ----------
    y : matrix (L, B)
        the target label sequences
    y_hat : tensor3 (T, B, C)
        class probabily distribution sequences
    y_hat_mask : matrix (T, B)
        indicates the lenghts of the sequences in y_hat
    Returns
    -------
    tensor3 (T, B, L):
        A tensor that contains the probabilities per time step of the
        labels that occur in the target sequence.
    """
    if y_hat_mask is not None:
        y_hat = y_hat * y_hat_mask[:, :, None]
    batch_size = y_hat.shape[1]
    y_hat = y_hat.dimshuffle(0, 2, 1)
    res = y_hat[:, y.astype('int32'), T.arange(batch_size)]
    return res.dimshuffle(0, 2, 1)


def _recurrence_relation(y, y_mask, blank_symbol):
    """
    Construct a permutation matrix and tensor for computing CTC transitions.

    This matrix is represented as an actual matrix that contains the
    permutations that are common to all transitions in the batch and a tensor
    with permutations that are uniqe for each individual sequence.

    This 'matrix' is used to take the transition costraints into account
    using just matrix algebra operations.

    Parameters
    ----------
    y : matrix (L, B)
        the target label sequences
    y_mask : matrix (L, B)
        indicates which values of y to use
    blank_symbol: integer
        indicates the symbol that signifies a blank label.
    Returns
    -------
    matrix (L, L)
    tensor3 (L, L, B)
    """
    n_y = y.shape[0]
    blanks = tensor.zeros((2, y.shape[1])) + blank_symbol
    ybb = tensor.concatenate((y, blanks), axis=0).T
    sec_diag = (tensor.neq(ybb[:, :-2], ybb[:, 2:]) *
                tensor.eq(ybb[:, 1:-1], blank_symbol) *
                y_mask.T)

    # r1: LxL
    # r2: LxL
    # r3: LxLxB
    eye2 = tensor.eye(n_y + 2)
    r2 = eye2[2:, 1:-1]  # tensor.eye(n_y, k=1)
    r3 = (eye2[2:, :-2].dimshuffle(0, 1, 'x') *
          sec_diag.dimshuffle(1, 'x', 0))

    return r2, r3


def _path_probabs(y, y_hat, y_mask, y_hat_mask, blank_symbol):
    """Compute the probabilities of the paths that are compatible with the
    sequence y.

    This function uses scan to get the forward probabilities (often denoted
    with the symbol alpha in the literature).

    See _log_path_probabs for a version that works in log domain.
    """




    pred_y = _class_batch_to_labeling_batch(y, y_hat, y_hat_mask)
    pred_y = pred_y.dimshuffle(0, 2, 1)
    n_labels = y.shape[0]

    r2, r3 = _recurrence_relation(y, y_mask, blank_symbol)

    def step(p_curr, p_prev):
        # instead of dot product, we * first
        # and then sum oven one dimension.
        # objective: T.dot((p_prev)BxL, LxLxB)
        # solusion: Lx1xB * LxLxB --> LxLxB --> (sumover)xLxB
        dotproduct = (p_prev + tensor.dot(p_prev, r2) +
                      (p_prev.dimshuffle(1, 'x', 0) * r3).sum(axis=0).T)
        return p_curr.T * dotproduct * y_mask.T  # B x L

    probabilities, _ = theano.scan(
        step,
        sequences=[pred_y],
        outputs_info=[tensor.eye(n_labels)[0] * tensor.ones(y.T.shape)])
    return probabilities


def _log_add(a, b):
    # TODO: move functions like this to utils
    max_ = tensor.maximum(a, b)
    result = (max_ + tensor.log1p(tensor.exp(a + b - 2 * max_)))
    return T.switch(T.isnan(result), max_, result)


def _log_dot_matrix(x, z):
    y = x[:, :, None] + z[None, :, :]
    y_max = y.max(axis=1)
    out = T.log(T.sum(T.exp(y - y_max[:, None, :]), axis=1)) + y_max
    return T.switch(T.isnan(out), -numpy.inf, out)


def _log_dot_tensor(x, z):
    log_dot = x.dimshuffle(1, 'x', 0) + z
    max_ = log_dot.max(axis=0)
    out = (T.log(T.sum(T.exp(log_dot - max_[None, :, :]), axis=0)) + max_)
    out = out.T
    return T.switch(T.isnan(out), -numpy.inf, out)


def _log_path_probabs(y, log_y_hat, y_mask, y_hat_mask, blank_symbol,
                     reverse=False):
    """
    Uses dynamic programming to compute the path probabilities.

    This function uses scan to get the forward probabilities (often denoted
    with the symbol alpha in the literature).

    This function computes the probabilities in log domain and can be used
    both the forward and backward passes of the CTC algorithm.

    Notes
    -----
    T: number of time steps
    B: batch size
    L: length of label sequence
    C: number of classes
    Parameters
    ----------
    y : matrix (L, B)
        the target label sequences
    log_y_hat : tensor3 (T, B, C)
        log class probabily distribution sequences
    y_mask : matrix (L, B)
        indicates which values of y to use
    y_hat_mask : matrix (T, B)
        indicates the lenghts of the sequences in log_y_hat
    blank_symbol: integer
        indicates the symbol that signifies a blank label.
    Returns
    -------
    tensor3 (T, B, L):
        the log forward probabilities for each label at every time step.
        masked values should be -inf
    """

    n_labels, batch_size = y.shape

    if reverse:
        y = y[::-1]
        log_y_hat = log_y_hat[::-1]
        y_hat_mask = y_hat_mask[::-1]
        y_mask = y_mask[::-1]
        # going backwards, the first non-zero alpha value should be the
        # first non-masked label.
        start_positions = T.cast(n_labels - y_mask.sum(0), 'int64')
    else:
        start_positions = T.zeros((batch_size,), dtype='int64')

    log_pred_y = _class_batch_to_labeling_batch(y, log_y_hat, y_hat_mask)
    log_pred_y = log_pred_y.dimshuffle(0, 2, 1)
    r2, r3 = _recurrence_relation(y, y_mask, blank_symbol)
    r2, r3 = T.log(r2), T.log(r3)

    def step(log_p_curr, y_hat_mask_t, log_p_prev):
        # applies the transitions matrices to take the sequence constraints of y
        # into account.
        p1 = log_p_prev
        p2 = _log_dot_matrix(p1, r2)
        p3 = _log_dot_tensor(p1, r3)
        p12 = _log_add(p1, p2)
        p123 = _log_add(p3, p12)

        y_hat_mask_t = y_hat_mask_t[:, None]
        out = log_p_curr.T + p123 + T.log(y_mask.T)
        return _log_add(T.log(y_hat_mask_t) + out,
                        T.log(1 - y_hat_mask_t) + log_p_prev)

    log_probabilities, _ = theano.scan(
        step,
        sequences=[log_pred_y, y_hat_mask],
        outputs_info=[T.log(tensor.eye(n_labels)[start_positions])])

    return log_probabilities + T.log(y_hat_mask[:, :, None])


def _log_forward_backward(y, log_y_hat, y_mask, y_hat_mask, blank_symbol):
    """Simply calls _log_path_probabs in both directions."""

    log_probabs_forward = _log_path_probabs(y,
                                            log_y_hat,
                                            y_mask,
                                            y_hat_mask,
                                            blank_symbol)
    log_probabs_backward = _log_path_probabs(y,
                                             log_y_hat,
                                             y_mask,
                                             y_hat_mask,
                                             blank_symbol,
                                             reverse=True)
    return log_probabs_forward, log_probabs_backward[::-1][:, :, ::-1]


def _labeling_batch_to_class_batch(y, y_labeling, num_classes):
    """Coverts a sequence label lattice into a lattice of scores/probabilities
    for each class per input time step.
    """

    batch_size = y.shape[1]
    N = y_labeling.shape[0]
    n_labels = y.shape[0]
    # sum over all repeated labels
    # from (T, B, L) to (T, C, B)
    out = T.zeros((num_classes, batch_size, N))
    y_labeling = y_labeling.dimshuffle((2, 1, 0))  # L, B, T
    y_ = y

    def scan_step(index, prev_res, y_labeling, y_):
        res_t = T.inc_subtensor(prev_res[y_[index, T.arange(batch_size)],
                                T.arange(batch_size)],
                                y_labeling[index, T.arange(batch_size)])
        return res_t

    result, updates = theano.scan(scan_step,
                                  sequences=[T.arange(n_labels)],
                                  non_sequences=[y_labeling, y_],
                                  outputs_info=[out])
    # result will be (C, B, T) so we make it (T, B, C)
    return result[-1].dimshuffle(2, 1, 0)
