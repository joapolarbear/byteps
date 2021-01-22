import tensorflow as tf
from google.protobuf.json_format import MessageToJson
import json
import networkx as nx
import struct, math
import numpy as np
import os, sys
from byteps.tensorflow.ops import local_rank, rank
from tensorflow.python.client import timeline
import threading

class _SecondOrStepTimer(tf.train.SecondOrStepTimer):
    def __init__(self, every_secs=None, every_steps=None, step_bound=None):
        if step_bound is not None:
            if not (isinstance(step_bound, list) or isinstance(step_bound, tuple)):
                raise ValueError("step bound must be a list or a tuple, but {} is given".format(step_bound))
            self._start_step = step_bound[0]
            self._end_step = step_bound[1]
            if self._start_step > self._end_step:
                raise ValueError("Profiling start step must be smaller than the end step.")
        else:
            self._start_step = self._end_step = None

        super(_SecondOrStepTimer, self).__init__(every_secs, every_steps)

    def should_trigger_for_step(self, step):
        if self._start_step is not None:
            if step < self._start_step or step > self._end_step:
                return False

        return super(_SecondOrStepTimer, self).should_trigger_for_step(step)

class TimelineHook(tf.train.ProfilerHook):
    def __init__(self, _summary=False, batch_size=None):
        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(local_rank()))
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)

        if os.environ.get("BYTEPS_TRACE_ON", "") != '1':
            self._end_trace = True
            self.start_step = self.end_step = 0
        else:
            self._end_trace = False
            self.start_step = int(os.environ.get("BYTEPS_TRACE_START_STEP", "20"))
            self.end_step = int(os.environ.get("BYTEPS_TRACE_END_STEP", "30"))
        
        if not self._end_trace and self.start_step < 1:
            raise ValueError("BYTEPS_TRACE_START_STEP must be larger than 1")
        if not self._end_trace and self.end_step <= self.start_step:
            raise ValueError("BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP")
        
        print("TimelineHook enable: {}  start_step: {} end_step: {}".format(not self._end_trace, self.start_step, self.end_step))
        self.dag = None
        self.has_data = False

        self._output_file = os.path.join(self.trace_dir, "timeline-{}.json")
        self._file_writer = tf.summary.FileWriterCache.get(self.trace_dir) if _summary else None
        self._show_dataflow = True
        self._show_memory = False
        self._timer = _SecondOrStepTimer(
            every_secs=None, every_steps=1, step_bound=(self.start_step, self.end_step))
        self.batch_size = batch_size
        assert self.batch_size is not None

    def before_run(self, run_context):
        if not self._end_trace:
            self._request_summary = (
                self._next_step is not None and
                self._timer.should_trigger_for_step(self._next_step))
            
            if self._request_summary and not self.has_data:
                ### the first step to collect traces, self.has_data tells there are data that need outputing
                self.has_data = True
            if self.has_data and not self._request_summary:
                ### the step after the last trace step, output data
                self._end_trace = True
                graphdef = tf.get_default_graph().as_graph_def(add_shapes=True)
                _t = threading.Thread(target=self.output_traces, args=(tf.get_default_graph().get_operations(), graphdef))
                _t.start()
        else:
            self._request_summary = False
                
        requests = {"global_step": self._global_step_tensor}
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            if self._request_summary else None)

        return tf.train.SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        if self._next_step is None:
        # Update the timer so that it does not activate until N steps or seconds
        # have passed.
            self._timer.update_last_triggered_step(stale_global_step)
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._timer.update_last_triggered_step(global_step)
            self._save(global_step, self._output_file.format(global_step),
                     run_values.run_metadata.step_stats)
            if self._file_writer is not None:
                self._file_writer.add_run_metadata(run_values.run_metadata,
                                         "step_%d" % global_step)
        self._next_step = global_step + 1

    def output_traces(self, ops, graphdef):
        self.traces = {"traceEvents":[]}    
        ### the ProfilerHook of tensorflow will output the timeline to self.trace_dir/timeline-{global_step}.json
        for file in sorted(os.listdir(self.trace_dir)):
            if file.startswith('timeline-'):
                with open(os.path.join(self.trace_dir, file), 'r') as fp:
                    ctf = json.load(fp)
                convert_traces = self.chome_trace_MBE2X(ctf["traceEvents"])
                self.traces["traceEvents"] += convert_traces 
        with open(os.path.join(self.trace_dir, "temp.json"), "w") as fp:
            json.dump(self.traces, fp, indent=4)

        if os.getenv("BYTEPS_PURE_TF_TRACE", '1') == '1':
            ### delete all intermediate redults
            _output_files = os.path.join(self.trace_dir, "timeline-*.json")
            os.system('rm {}'.format(_output_files))

        def serialize_tensor(t):
            _shape = t.shape.as_list() if t.shape.dims is not None else []
            if len(_shape) > 0 and _shape[0] is None:
                _shape[0] = self.batch_size
            return {
                "name": t.name,
                "shape": _shape,
                "dtype": t.dtype.name
            }

        if rank() == 0:
            ### Only dump these info for rank 0   
            op_dict = {}
            for op in ops:
                op_dict[op.name] = {
                    "output":[serialize_tensor(e) for e in op.outputs],
                    "input": [serialize_tensor(e) for e in op.inputs._inputs],
                    "op": op.type
                }
            with open(os.path.join(self.trace_dir, "metadata.json"), "w") as f:
                json.dump(op_dict, f, indent=4)

            if self.dag is not None:
                nx.write_gml(self.dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))
            
            self.add_infer_shape_ops(op_dict, graphdef)

        print("Stop tracing, output trace at %s" % self.trace_dir)

    def chome_trace_MBE2X(self, raw_traces):
        ret = []
        pid_table = {}
        if self.dag is None:
            _dag = nx.DiGraph()
        for trace in raw_traces:
            ### Create the DAG
            if self.dag is None:
                if trace["ph"] == "M" or "args" not in trace:
                    continue
                op = trace["args"]["op"]
                name = trace["args"]["name"]
                if name.startswith("^"):
                    name = name[1:]
                ### Add dependency info
                for k, v in trace["args"].items():
                    if "input" in k:
                        if v.startswith("^"):
                            v = v[1:]
                        _dag.add_edge(v, name)
                    
            if trace["ph"] == "M":
                if trace["name"] == "process_name":
                    assert trace["pid"] not in pid_table
                    if trace["args"]["name"] == "":
                        continue
                    process_name = trace["args"]["name"]
                    if "stream:all Compute" in process_name and "device:GPU" in process_name:
                        pid_table[trace["pid"]] = {"process_name": process_name}
                else:
                    pass
            elif trace["ph"] == "i":
                trace["pid"] = trace["tid"] = "mark"
                ret.append(trace)
            elif trace["pid"] in pid_table and trace["ph"] == "X":
                cur_pid = pid_table[trace["pid"]]
                trace["pid"] = cur_pid["process_name"]
                ret.append(trace)
            else:
                pass
        if self.dag is None:
            self.dag = _dag
        return ret
    
    ### TODO (huhanpeng) need to be cleaned up
    def add_infer_shape_ops(self, op_dict, graphdef):
        self.tensor_shapes = {}
        for name in op_dict.keys():
            self.tensor_shapes[name] = []
            for shape_dict in op_dict[name]["output"]:
                if name in shape_dict["name"]:
                    self.tensor_shapes[name] = shape_dict["shape"]
                    break
        with open(os.path.join(self.trace_dir, "tensor_shapes.json"), "w") as f:
            json.dump(self.tensor_shapes, f, indent=4)
        
        graph_str = json.loads(MessageToJson(graphdef))
        with open(os.path.join(self.trace_dir, "final_graph.json"), "w") as f:
            json.dump(graph_str, f, indent=4)
        
        # with open(os.path.join(self.trace_dir, "final_graph.json"), "r") as f:
        #     graph_def_as_json = json.load(f)
        # cleaned_graph_def_str = json.dumps(graph_def_as_json)
        # from google.protobuf.json_format import Parse
        # graph_def = Parse(cleaned_graph_def_str, tf.GraphDef())
        # ## collect graph info
        # self.original_graph = tf.Graph()
        # with self.original_graph.as_default():
        #     tf.import_graph_def(graph_def, name="")
        # print(self.original_graph.get_operations())

