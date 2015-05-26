"""
CTC-Connectionist Temporal Classification

Code provided by Mohammad Pezeshki - May. 2015 -
Montreal Institute for Learning Algorithms

Referece: Graves, Alex, et al. "Connectionist temporal classification:
labelling unsegmented sequence data with recurrent neural networks."
Proceedings of the 23rd international conference on Machine learning.
ACM, 2006.

Credits: Shawn Tan, Rakesh Var

This code is distributed without any warranty, express or implied.
"""

import theano
from theano import tensor

floatX = theano.config.floatX


# T: INPUT_SEQUENCE_LENGTH
# B: BATCH_SIZE
# L: OUTPUT_SEQUENCE_LENGTH
# C: NUM_CLASSES
class CTC(object):
    """Connectionist Temporal Classification
    y_hat : T x B x C+1
    y : L x B
    y_hat_mask : T x B
    y_mask : L x B
    """
    @staticmethod
    def add_blanks(y, blank_symbol, y_mask=None):
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
        return blanked_y, blanked_y_mask

    @staticmethod
    def class_batch_to_labeling_batch(y, y_hat, y_hat_mask=None):
        y_hat = y_hat * y_hat_mask.dimshuffle(0, 'x', 1)
        batch_size = y_hat.shape[2]
        res = y_hat[:, y.astype('int32'), tensor.arange(batch_size)]
        return res

    @staticmethod
    def recurrence_relation(y, y_mask, blank_symbol):
        n_y = y.shape[0]
        blanks = tensor.zeros((2, y.shape[1])) + blank_symbol
        ybb = tensor.concatenate((y, blanks), axis=0).T
        sec_diag = (tensor.neq(ybb[:, :-2], ybb[:, 2:]) *
                    tensor.eq(ybb[:, 1:-1], blank_symbol) *
                    y_mask.T)

        # r1: LxL
        # r2: LxL
        # r3: LxLxB
        r2 = tensor.eye(n_y, k=1)
        r3 = (tensor.eye(n_y, k=2).dimshuffle(0, 1, 'x') *
              sec_diag.dimshuffle(1, 'x', 0))

        return r2, r3

    @classmethod
    def path_probabs(cls, y, y_hat, y_mask, y_hat_mask, blank_symbol):
        pred_y = cls.class_batch_to_labeling_batch(y, y_hat, y_hat_mask)

        r2, r3 = cls.recurrence_relation(y, y_mask, blank_symbol)

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
            outputs_info=[tensor.eye(y.shape[0])[0] * tensor.ones(y.T.shape)])
        return probabilities, probabilities.shape

    @classmethod
    def cost(cls, y, y_hat, y_mask, y_hat_mask, blank_symbol):
        y_hat_mask_len = tensor.sum(y_hat_mask, axis=0, dtype='int32')
        y_mask_len = tensor.sum(y_mask, axis=0, dtype='int32')
        probabilities, sth = cls.path_probabs(y, y_hat,
                                              y_mask, y_hat_mask,
                                              blank_symbol)
        batch_size = probabilities.shape[1]
        labels_probab = (probabilities[y_hat_mask_len - 1,
                                       tensor.arange(batch_size),
                                       y_mask_len - 1] +
                         probabilities[y_hat_mask_len - 1,
                                       tensor.arange(batch_size),
                                       y_mask_len - 2])
        avg_cost = tensor.mean(-tensor.log(labels_probab))
        return avg_cost, sth

    @staticmethod
    def _epslog(x):
        return tensor.cast(tensor.log(tensor.clip(x, 1E-12, 1E12)),
                           theano.config.floatX)

    @staticmethod
    def log_add(a, b):
        max_ = tensor.maximum(a, b)
        return (max_ + tensor.log1p(tensor.exp(a + b - 2 * max_)))

    @staticmethod
    def log_dot_matrix(x, z):
        inf = 1E12
        log_dot = tensor.dot(x, z)
        zeros_to_minus_inf = (z.max(axis=0) - 1) * inf
        return log_dot + zeros_to_minus_inf

    @staticmethod
    def log_dot_tensor(x, z):
        inf = 1E12
        log_dot = (x.dimshuffle(1, 'x', 0) * z).sum(axis=0).T
        zeros_to_minus_inf = (z.max(axis=0) - 1) * inf
        return log_dot + zeros_to_minus_inf.T

    @classmethod
    def log_path_probabs(cls, y, y_hat, y_mask, y_hat_mask, blank_symbol):
        pred_y = cls.class_batch_to_labeling_batch(y, y_hat, y_hat_mask)
        r2, r3 = cls.recurrence_relation(y, y_mask, blank_symbol)

        def step(log_p_curr, log_p_prev):
            p1 = log_p_prev
            p2 = cls.log_dot_matrix(p1, r2)
            p3 = cls.log_dot_tensor(p1, r3)
            p123 = cls.log_add(p3, cls.log_add(p1, p2))

            return (log_p_curr.T +
                    p123 +
                    cls._epslog(y_mask.T))

        log_probabilities, _ = theano.scan(
            step,
            sequences=[cls._epslog(pred_y)],
            outputs_info=[cls._epslog(tensor.eye(y.shape[0])[0] *
                                      tensor.ones(y.T.shape))])
        return log_probabilities

    @classmethod
    def log_cost(cls, y, y_hat, y_mask, y_hat_mask, blank_symbol):
        y_hat_mask_len = tensor.sum(y_hat_mask, axis=0, dtype='int32')
        y_mask_len = tensor.sum(y_mask, axis=0, dtype='int32')
        log_probabs = cls.log_path_probabs(y, y_hat,
                                           y_mask, y_hat_mask,
                                           blank_symbol)
        batch_size = log_probabs.shape[1]
        labels_probab = cls.log_add(
            log_probabs[y_hat_mask_len - 1,
                        tensor.arange(batch_size),
                        y_mask_len - 1],
            log_probabs[y_hat_mask_len - 1,
                        tensor.arange(batch_size),
                        y_mask_len - 2])
        avg_cost = tensor.mean(-labels_probab)
        return avg_cost

    @classmethod
    def apply(cls, y, y_hat, y_mask, y_hat_mask, scale='log_scale'):
        y_hat = y_hat.dimshuffle(0, 2, 1)
        num_classes = y_hat.shape[1] - 1
        blanked_y, blanked_y_mask = cls.add_blanks(
            y=y,
            blank_symbol=num_classes.astype(floatX),
            y_mask=y_mask)
        if scale == 'log_scale':
            final_cost = cls.log_cost(blanked_y, y_hat,
                                      blanked_y_mask, y_hat_mask,
                                      num_classes)
        else:
            final_cost, sth = cls.cost(blanked_y, y_hat,
                                       blanked_y_mask, y_hat_mask,
                                       num_classes)
        return final_cost
