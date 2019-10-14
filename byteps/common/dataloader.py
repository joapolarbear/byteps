import os
import time
import json
import threading

class IORecorder(object):
    def __init__(self):
        if os.environ.get("BYTEPS_TRACE_ON", "") == "1":
            self._end_trace = True
        self._end_trace = False
        self.trace_dir = os.environ.get("BYTEPS_TRACE_DIR", ".") + "/" + os.environ.get("BYTEPS_LOCAL_RANK") + "/"
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)
        self.trace_path = self.trace_dir + 'io.json'
        self.ts = []
        self.dur = []

    def io_start(self):
        if self._end_trace:
            return
        if os.environ.get("BYTEPS_TRACE_STATUS", "") == "END":
            self._end_trace = True
            self.output_traces()
            return
        self.ts.append(time.time() * 1000000.0)

    def io_end(self):
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
                    "name": "I/O",
                    "ts": _ts,
                    "dur": _dur,
                    "ph": "X",
                    "cat": "I/O",
                    "pid": "I/O",
                    "args": {
                        "name":"I/O"
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

class BPSMultiWorkerIter(object):
    def __init__(self, data_iter):
        self._data_iter = data_iter
        self.recorder = IORecorder()

    def _push_next_dataset(self):
        self._data_iter._push_next_dataset()

    def _push_next(self):
        self._data_iter._push_next()

    def _next_dataset(self):
        return self._data_iter._next_dataset()

    def __next__(self):
        self.recorder.io_start()
        ret = self._data_iter.__next__()
        self.recorder.io_end()
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






