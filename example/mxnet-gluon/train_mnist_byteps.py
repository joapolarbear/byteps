# Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file is modified from `horovod/examples/mxnet_mnist.py`, using gluon style MNIST dataset and data_loader."""
import time

import argparse
import logging

import mxnet as mx
import byteps.mxnet as bps
from mxnet import autograd, gluon, nd
from mxnet.gluon.data.vision import MNIST

import os
from common import data_byteps


# Higher download speed for chinese users
# os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'

# Training settings
parser = argparse.ArgumentParser(description='MXNet MNIST Example')

parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 64)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='training data type (default: float32)')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs (default: 5)')
parser.add_argument('--j', type=int, default=2,
                    help='number of cpu processes for dataloader')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable training on GPU (default: False)')

parser.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
parser.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
parser.add_argument('--image-shape', type=str, default='1,28,28',
                      help='the image shape feed into the network, e.g. (3,224,224)')
parser.add_argument('--benchmark', type=int, default=1,
                      help='if 1, then feed the network with synthetic data')
parser.add_argument('--num-classes', type=int, default=10,
                        help='the number of classes')
parser.add_argument('--num-examples', type=int, default=1000000,
                        help='the number of training examples')


args = parser.parse_args()

if not args.no_cuda:
    # Disable CUDA if there are no GPUs.
    if mx.context.num_gpus() == 0:
        args.no_cuda = True

logging.basicConfig(level=logging.INFO)
logging.info(args)


def dummy_transform(data, label):
    im = data.astype(args.dtype, copy=False) / 255 - 0.5
    im = nd.transpose(im, (2, 0, 1))
    return im, label


# Function to get mnist iterator
def get_mnist_iterator():
    train_set = MNIST(train=True, transform=dummy_transform)
    train_iter = gluon.data.DataLoader(train_set, args.batch_size, True, num_workers=args.j, last_batch='discard')
    val_set = MNIST(train=False, transform=dummy_transform)
    val_iter = gluon.data.DataLoader(val_set, args.batch_size, False, num_workers=0)

    return train_iter, val_iter, len(train_set)


# Function to define neural network
def conv_nets():
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(10))
    return net


# Function to evaluate accuracy for a model
def evaluate(model, data_iter, context):
    metric = mx.metric.Accuracy()
    for _, batch in enumerate(data_iter):
        data = batch[0].as_in_context(context)
        label = batch[1].as_in_context(context)
        output = model(data.astype(args.dtype, copy=False))
        metric.update([label], [output])

    return metric.get()


# Load training and validation data
train_data, val_data, train_size = get_mnist_iterator()

# Initialize BytePS
bps.init()

# BytePS: pin context to local rank
context = mx.cpu(bps.local_rank()) if args.no_cuda else mx.gpu(bps.local_rank())
num_workers = bps.size()

# Build model
model = conv_nets()
model.cast(args.dtype)

# Initialize parameters
model.initialize(mx.init.MSRAPrelu(), ctx=context)
# if bps.rank() == 0:
model.summary(nd.ones((1, 1, 28, 28), ctx=mx.gpu(bps.local_rank())))
model.hybridize()
# Create loss function and train metric
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()

# --------------------- warmup and export ---------------
batch = list(train_data)[0]
data = batch[0].as_in_context(context)
label = batch[1].as_in_context(context)
output = model(data)
prefix = "GluonModel"
model.export(prefix)
assert os.path.isfile(prefix + '-symbol.json')
assert os.path.isfile(prefix + '-0000.params')

# --------------------- import with SymbolBlock ----------
# imported_net = mx.gluon.nn.SymbolBlock.imports(prefix + '-symbol.json',
#                                                    ['data'],
#                                                    prefix + '-0000.params',
#                                                    ctx=context)
imported_net = model._cached_graph[1]

# BytePS: create DistributedTrainer, a subclass of gluon.Trainer
optimizer_params = {'momentum': args.momentum, 'learning_rate': args.lr * num_workers}

# BytePS: fetch and broadcast parameters
# params = imported_net.collect_params()
# trainer = bps.DistributedTrainer(params, "sgd", optimizer_params, block=imported_net)

# '''
# --------------------- create new symbol module based on the imported net ----------
network = mx.sym.SoftmaxOutput(data=imported_net, name='softmax')
sym_model = mx.mod.Module(
        context=context,
        symbol=network
    )

eval_metrics = ['accuracy']
batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
opt = mx.optimizer.create(args.optimizer, sym=network, **optimizer_params)
opt = bps.DistributedOptimizer(opt, sym=network)

(train, val) = data_byteps.get_rec_iter(args, (bps.rank(), bps.size()))
sym_model.bind(data_shapes=train.provide_data,
           label_shapes=train.provide_label)

initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
sym_model.init_params(initializer)
arg_params, aux_params = sym_model.get_params()
if arg_params is not None:
    bps.broadcast_parameters(arg_params, root_rank=0)
if aux_params is not None:
    bps.broadcast_parameters(aux_params, root_rank=0)
sym_model.set_params(arg_params=arg_params, aux_params=aux_params)


sym_model.fit(train,
      begin_epoch=0,
      num_epoch=args.epochs,
      eval_data=val,
      eval_metric=eval_metrics,
      kvstore=None,
      optimizer=opt,
      optimizer_params=optimizer_params,
      batch_end_callback=batch_end_callbacks,
      epoch_end_callback=None,
      allow_missing=True,
      monitor=None)
# '''
'''
# Train model
for epoch in range(args.epochs):
    tic = time.time()
    metric.reset()
    for i, batch in enumerate(train_data):
        data = batch[0].as_in_context(context)
        label = batch[1].as_in_context(context)

        with autograd.record():
            output = imported_net(data)
            loss = loss_fn(output, label)

        loss.backward()
        trainer.step(args.batch_size)
        metric.update([label], [output])

        if i % 100 == 0:
            name, acc = metric.get()
            logging.info('[Epoch %d Batch %d] Training: %s=%f' %
                         (epoch, i, name, acc))

    if bps.rank() == 0:
        elapsed = time.time() - tic
        speed = train_size * num_workers / elapsed
        logging.info('Epoch[%d]\tSpeed=%.2f samples/s\tTime cost=%f',
                     epoch, speed, elapsed)

    # Evaluate model accuracy
    _, train_acc = metric.get()
    name, val_acc = evaluate(imported_net, val_data, context)
    if bps.rank() == 0:
        logging.info('Epoch[%d]\tTrain: %s=%f\tValidation: %s=%f', epoch, name,
                     train_acc, name, val_acc)

    if bps.rank() == 0 and epoch == args.epochs - 1:
        assert val_acc > 0.96, "Achieved accuracy (%f) is lower than expected\
                                (0.96)" % val_acc
'''
