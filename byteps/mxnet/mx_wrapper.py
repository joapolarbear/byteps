import os
import time
import json
import threading

class TimtLineRecorder(object):
    def __init__(self, _trace_name, _name):
        if os.environ.get("BYTEPS_TRACE_ON", "") == "1":
            self._end_trace = True
        self._end_trace = False
        self.trace_dir = os.environ.get("BYTEPS_TRACE_DIR", ".") + "/" + os.environ.get("BYTEPS_LOCAL_RANK") + "/"
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)
        self.trace_path = self.trace_dir + _trace_name
        self.ts = []
        self.dur = []
        self._name = _name

    def start(self):
        if self._end_trace:
            return
        if os.environ.get("BYTEPS_TRACE_STATUS", "") == "END":
            self._end_trace = True
            self.output_traces()
            return
        self.ts.append(time.time() * 1000000.0)

    def end(self):
        if self._end_trace:
            return
        assert len(self.ts) == len(self.dur) + 1 or len(self.ts) == len(self.dur)
        if len(self.ts) == len(self.dur) + 1:
            self.dur.append(time.time() * 1000000.0 - self.ts[-1])

    def output_traces(self):
        def _output(self):
            rst_traces = {"traceEvents": []}
            for i in range(len(self.dur)):
                _ts, _dur = self.ts[i], self.dur[i]
                _event = {
                    "name": self._name,
                    "ts": _ts,
                    "dur": _dur,
                    "ph": "X",
                    "cat": self._name,
                    "pid": self._name,
                    "args": {
                        "name":self._name
                    }
                }
                rst_traces["traceEvents"].append(_event)
            rst_traces["displayTimeUnit"] = "ms"
            with open(self.trace_path, 'w') as f:
                json.dump(rst_traces, f, indent=4)
            self.ts = []
            self.dur = []
        t = threading.Thread(target=_output, args=(self,))
        t.start()

########################################################
#      Used to wrap IO iterator
########################################################

class BPSMultiWorkerIter(object):
    def __init__(self, data_iter):
        self._data_iter = data_iter
        self.recorder = TimtLineRecorder('io.json', 'I/O')

    def _push_next_dataset(self):
        self._data_iter._push_next_dataset()

    def _push_next(self):
        self._data_iter._push_next()

    def _next_dataset(self):
        return self._data_iter._next_dataset()

    def __next__(self):
        self.recorder.start()
        ret = self._data_iter.__next__()
        self.recorder.end()
        return ret

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __len__(self):
        return self._data_iter.__len__()

class BPSDatasetLoader(object):
    def __init__(self, dataloader):
        self._dataloader = dataloader

    def __iter__(self):
        return BPSMultiWorkerIter(self._dataloader.__iter__())

    def __len__(self):
        return self._dataloader.__len__()


########################################################
#       Trainer, used to wrap the process
#            of updating local model
########################################################

class BPSTrainer(object):
    def __init__(self, _trainer):
        self._trainer = _trainer
        self.recorder = TimtLineRecorder('step.json', 'STEP')

    def backward(self, *args, **kwargs):
        self._trainer.backward(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.recorder.start()
        self._trainer.step(*args, **kwargs)
        self.recorder.end()



