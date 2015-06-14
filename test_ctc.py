import theano
import numpy
from theano import tensor
from blocks.model import Model
from blocks.bricks import Linear, Tanh
from ctc_cost import CTC
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.graph import ComputationGraph
try:
    import cPickle as pickle
except:
    import pickle

floatX = theano.config.floatX


@theano.compile.ops.as_op(itypes=[tensor.lvector],
                          otypes=[tensor.lvector])
def print_pred(y_hat):
    blank_symbol = 4
    res = []
    for i, s in enumerate(y_hat):
        if (s != blank_symbol) and (i == 0 or s != y_hat[i - 1]):
            res += [s]
    return numpy.asarray(res)

n_epochs = 200
x_dim = 4
h_dim = 9
num_classes = 4

with open("ctc_test_data.pkl", "rb") as pkl_file:
    try:
        data = pickle.load(pkl_file)
        inputs = data['inputs']
        labels = data['labels']
        # from S x T x B x D to S x T x B
        inputs_mask = numpy.max(data['mask_inputs'], axis=-1)
        labels_mask = data['mask_labels']
    except:
        data = pickle.load(pkl_file, encoding='bytes')
        inputs = data[b'inputs']
        labels = data[b'labels']
        # from S x T x B x D to S x T x B
        inputs_mask = numpy.max(data[b'mask_inputs'], axis=-1)
        labels_mask = data[b'mask_labels']



print('Building model ...')
# T x B x F
x = tensor.tensor3('x', dtype=floatX)
# T x B
x_mask = tensor.matrix('x_mask', dtype=floatX)
# L x B
y = tensor.matrix('y', dtype=floatX)
# L x B
y_mask = tensor.matrix('y_mask', dtype=floatX)

x_to_h = Linear(name='x_to_h',
                input_dim=x_dim,
                output_dim=h_dim)
x_transform = x_to_h.apply(x)
rnn = SimpleRecurrent(activation=Tanh(),
                      dim=h_dim, name="rnn")
h = rnn.apply(x_transform)
h_to_o = Linear(name='h_to_o',
                input_dim=h_dim,
                output_dim=num_classes + 1)
h_transform = h_to_o.apply(h)
# T x B x C+1
y_hat = tensor.nnet.softmax(
    h_transform.reshape((-1, num_classes + 1))
).reshape((h.shape[0], h.shape[1], -1))
y_hat.name = 'y_hat'

y_hat_mask = x_mask
cost = CTC().apply(y, y_hat, y_mask, y_hat_mask, 'normal_scale')
cost.name = 'CTC'
# Initialization
for brick in (rnn, x_to_h, h_to_o):
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0)
    brick.initialize()

print('Bulding DataStream ...')
dataset = IterableDataset({'x': inputs,
                           'x_mask': inputs_mask,
                           'y': labels,
                           'y_mask': labels_mask})
stream = DataStream(dataset)

print('Bulding training process...')
algorithm = GradientDescent(cost=cost,
                            params=ComputationGraph(cost).parameters,
                            step_rule=CompositeRule([StepClipping(10.0),
                                                     Scale(0.02)]))
monitor_cost = TrainingDataMonitoring([cost],
                                      prefix="train",
                                      after_epoch=True)

# sample number to monitor
sample = 8

y_hat_max_path = print_pred(tensor.argmax(y_hat[:, sample, :], axis=1))
y_hat_max_path.name = 'Viterbi'
monitor_output = TrainingDataMonitoring([y_hat_max_path],
                                        prefix="y_hat",
                                        every_n_epochs=1)

length = tensor.sum(y_mask[:, sample]).astype('int32')
tar = y[:length, sample].astype('int32')
tar.name = '_Target_Seq'
monitor_target = TrainingDataMonitoring([tar],
                                        prefix="y",
                                        every_n_epochs=1)

model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                     extensions=[monitor_cost, monitor_output,
                                 monitor_target,
                                 FinishAfter(after_n_epochs=n_epochs),
                                 Printing()],
                     model=model)

print('Starting training ...')
main_loop.run()
