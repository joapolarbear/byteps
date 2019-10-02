

### Modified files
* byteps/mxnet/__init__.py
* byteps/mxnet/ops.py
* byteps/mxnet/ops.cc
* byteps/mxnet/ops.h
* byteps/common/operations.cc
* byteps/common/operations.h
* byteps/common/core_loops.cc
* byteps/common/common.h

### Other files
* example/mxnet/train_imagenet_byteps.py
* example/mxnet/common/fit_byteps.py: **Add an argument for `DistributedOptimizer`**: `_opt = bps.DistributedOptimizer(opt, sym=network)`