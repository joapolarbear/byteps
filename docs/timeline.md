# Performance Analysis of BytePS

You can analyze the performance with the timeline profiled by the servers. It shows the processing duration (from the arrival time to the finish time) of each push/pull request from differnet workers. You might be able to find out the straggler who slows down the training. 


## Usage for Servers
Use `export BYTEPS_SERVER_ENABLE_PROFILE=1` to enable the profiling (only valid for servers).

Each server will generate a `server_profile.json` file in current directory. You can also specify the file name and location with `export BYTEPS_SERVER_PROFILE_OUTPUT_PATH=/path/to/your/server_profile.json`

By default it profiles requests of all keys (tensors). Therefore, you may find your `server_profile.json` too large and difficult to analyze. Instead, you can select a specific key ID (from the original `server_profile.json`) to profile: `export BYTEPS_SERVER_KEY_TO_PROFILE=KEY_ID`. For example, if you set `BYTEPS_SERVER_KEY_TO_PROFILE` to `27000832` you will get the following results:

```
......
{"name": "push-9", "ph": "B", "pid": 27000832, "tid": 27000832, "ts": 1569331999234742},
{"name": "push-9", "ph": "E", "pid": 27000832, "tid": 27000832, "ts": 1569331999234778},
{"name": "pull-9", "ph": "B", "pid": 27000832, "tid": 27000832, "ts": 1569331999234878},
{"name": "push-11", "ph": "B", "pid": 27000832, "tid": 27000832, "ts": 1569331999234898},
{"name": "push-11", "ph": "E", "pid": 27000832, "tid": 27000832, "ts": 1569331999234931},
......
```

Then analyze the timeline using `chrome://tracing`.
For example, below shows the profile result of a distributed training case (2 workers and 2 servers). In ps-lite, worker ranks are 9, 11, 13, and etc. So `push-9` and `push-11`  mean the push requests from the first worker and second worker, respectively. From this figure, we can observe that the first worker is slower than the second one. Similarly, you can find whether there is a consistent straggler for large scale training.
![profile](https://user-images.githubusercontent.com/13852819/65565724-53bb3b80-df83-11e9-8490-6bb590d6fd18.png)

---


## Usage For Workers

Use the following environment variables to enable profiling the operations runing on workers, including computation, communication operations: 

``` python
"BYTEPS_TRACE_ON" = "1"
"BYTEPS_TRACE_END_STEP" = "20"
"BYTEPS_TRACE_START_STEP"="10"
"BYTEPS_TRACE_DIR"= "./traces"
```

First `BYTEPS_TRACE_ON` should be set to `1` to enable profiling communication traces. `BYTEPS_TRACE_START_STEP` and `BYTEPS_TRACE_END_STEP` decides the step interval we want to profile, traces from step `BYTEPS_TRACE_START_STEP` to step `BYTEPS_TRACE_END_STEP` steps will be automatically collected and the result traces will be output in the chrome trace format. `BYTEPS_TRACE_DIR` denotes the path you want to store traces. 

Besides, when using the `bps.DistributedTrainer()` in your program, two additional arguments should be given: 1) `block`, class `mxnet.gluon.HybridBlock`, the model to train and has called `hybridize()`. 2) `loss`, a list of `mxnet.gluon.Loss`, the loss of the model, each of which must has called `hybridize()`. Below shows an example.

```python
trainer = bps.DistributedTrainer(param_dict, args.optimizer, optim_params, block=model.bert, loss=[None, None, model.nsp_loss, model.mlm_loss])
```
Note that a model may have multiple outputs and multiple loss nodes, here the loss node list must correspond to the order of outputs used for the loss. The order can be found in the first few lines of `block.debug_str()` . If the list is empty or all of the elements are `None`, loss nodes will be ignored.

To further collect I/O operations, you should wrap your `DataLoader` with `byteps.common.dataloader`, below shows an example,

```python
from byteps.common.dataloader import BPSDatasetLoader
data_train = BPSDatasetLoader(data_train)
data_train_iter = iter(data_train)
```

The result directory is organized as follows. 
``` tex
traces/
├── 0
│   ├── bps_trace_local_rank0_20step.json
│   ├── comm.json
│   ├── dag.gml
│   ├── io.json
│   ├── symbol_debug_str.txt
│   ├── loss2.txt
│   ├── loss3.txt
│   └── temp.json
└── 1
    ├── bps_trace_local_rank1_20step.json
    ├── comm.json
    ├── dag.gml
    ├── io.json
    ├── symbol_debug_str.txt
    └── temp.json
```

Here, `traces/` is the trace directory we defined using `BYTEPS_TRACE_DIR`. `traces/` contains several sub-directories, each of which denotes one GPU and is named with the local rank of this GPU, e.g., path `./traces/0/` stores the traces results of the GPU whose local rank is `0`. Each sub-directory contains following directories/files:
* `comm.json`: the final trace file containing the communication traces of all gradients;
* `io.json`: the final trace file containing the I/O traces;
* `temp.json`: a JSON file dumped using MXNet profiler, containing all computation traces;
* **`bps_trace_local_rank0_20step.json`**: the final trace file which combines computation, communication and I/O traces;
* `symbol_debug_str.txt`:  A file containing the model symbol information (`block.debug_str()`).
* `loss2.txt`:  A file containing the loss symbol information, here `2` is the index of output used by the loss function. 
* `dag.gml`: a completed graph which contains `FW (forward)`, `BW (backward)`, `Comm (communication)`, and `I/O ` nodes, besides special nodes like `OUTPUT` and `Sync` are also included. `FW` ends with `OUTPUT` and `BW` starts with `OUTPUT`. All `Comm` nodes is forced to connected with the `Sync` node, instead of original respective `FW` nodes, otherwise, the Graph would not be a DAG.

### Visualization

All these JSON files can be visualized using `chrome://tracing`.

Below shows a visualization example of `comm.json`. 
<img src="https://user-images.githubusercontent.com/17765864/69711658-634e3080-113c-11ea-8d70-fb75f89f2791.png" width="1916">

### Trace Analysis

Please refer to [here](https://github.com/joapolarbear/byteprofile-analysis) for more details about trace analysis.

### Trace Format

Let's look deep into the trace format.
``` json
{
    "ph": "X",
    "args": {
        "name": "Comm.byteps.gradient_0"
    },
    "pid": "Comm.byteps.gradient_0",
    "name": "Comm.byteps.gradient_0",
    "ts": 1574685989504865,
    "dur": 24026,
    "tid": "total"
},
{
    "ph": "X",
    "args": {
        "name": "Comm.byteps.gradient_0"
    },
    "pid": "Comm.byteps.gradient_0",
    "name": "Comm.byteps.gradient_0.BROADCAST",
    "ts": 1574685984662375,
    "dur": 1074,
    "tid": "26148864"
}
```
Basically, the trace event format is the same as the standard [Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit). Here, `name` is the name of one event, which can be shown on `chrome://tracing/`. Considering BytePS divides each gradinets to multiple partitions if necessary and each partition needs to go through several types of following operations, namely `QueueType`.
```
  "COORDINATE_REDUCE",
  "REDUCE",
  "COPYD2H",
  "PCIE_REDUCE",
  "COORDINATE_PUSH",
  "PUSH",
  "PULL",
  "COPYH2D",
  "COORDINATE_BROADCAST",
  "BROADCAST"
```
So there are two types of events:
1. If `tid` is `total`, the event records the entire interval to synchronize one gradient, including the queue time. In this case, `name` ends with the gradient index. 
2. If `tid` is a number, the event records the interval for each `QueueType` of each partition of one gradient. In this case, `name` ends with the gradient index and the corresponding `QueueType`, `tid` denotes the partition id.

Note that for BytePS, for multiple GPUs on one worker, only the root GPU is responsible for synchronizing with servers, and these GPUs located on one worker update parameters through all-reduce. Therefore, you can observe `PUSH` and `PULL` operations only in the traces of the root GPU. By default, the root GPU is one with the largest local rank.

### Overhead
Below shows the latency when running [`bert_12_768_12`](https://github.com/joapolarbear/gluon-nlp/tree/bert-byteprofile/scripts/bert) model with 2 workers, each containing 2 V100 GPUs with 16GB of memory. BytePS Timeline collects traces during step 10 to step 20 and after step 20, it asynchronously outputs the trace results, which may also cause extra overhead. Ignoring the warm up time (the first 10 steps), the overhead induced by BytePS Timeline is small. 
<img src="" width="1916">
