# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
"""Functions for adding delays for computation nodes."""
import logging

from mxnet import symbol
from mxnet.symbol import Symbol
from mxnet.symbol import contrib as symbol_contrib
from mxnet import ndarray
from mxnet.ndarray import NDArray
from mxnet.contrib.amp import lists
from mxnet import base

import time
import os

class Delayer:
    def __init__(self):
        _delay = os.getenv("BYTEPS_TRACE_DELAY_CMP", None)
        if _delay is None:
            return
        self.SLEEP_TIME = float(_delay) / 1000.0
        self._initialized = False
        if not self._initialized:
            self._initialized = True
            logging.info("Using synthetic delays.")
            self._wrap_symbol_functions(symbol)
            self._wrap_symbol_functions(ndarray)
            self._wrap_loss_output_functions(ndarray)
            self._wrap_loss_output_functions(symbol)

    def _get_fun_to_wrap(self, name, module, submodule_dict):
        module_internal = getattr(module, "_internal")
        prefix = base._get_op_name_prefix(name)
        if len(prefix) > 0:
            if prefix != '_random_' or name.endswith('_like'):
                func_name = name[len(prefix):]
                cur_module = submodule_dict[prefix]
            else:
                func_name = name
                cur_module = module_internal
        elif name.startswith('_'):
            func_name = name
            cur_module = module_internal
        else:
            func_name = name
            cur_module = module
        return func_name, cur_module

    def _wrap_symbol_functions(self, module):
        def _ndarray_wrapper(f):
            def _new_fun(*args, **kwargs):
                time.sleep(self.SLEEP_TIME)
                return f(*args, **kwargs)
            _new_fun.__name__ = f.__name__
            _new_fun.__module__ = f.__module__
            _new_fun.__doc__ = f.__doc__
            return _new_fun

        def _symbol_wrapper(f):
            def _new_fun(*args, **kwargs):
                time.sleep(self.SLEEP_TIME)
                return f(*args, **kwargs)
            _new_fun.__name__ = f.__name__
            _new_fun.__module__ = f.__module__
            _new_fun.__doc__ = f.__doc__
            return _new_fun

        def _symbol_widest_wrapper(f):
            def _new_fun(*args, **kwargs):
                time.sleep(self.SLEEP_TIME)
                return f(*args, **kwargs)
            _new_fun.__name__ = f.__name__
            _new_fun.__module__ = f.__module__
            _new_fun.__doc__ = f.__doc__
            return _new_fun

        _wrapper = _symbol_wrapper if module in (symbol, Symbol, symbol_contrib) else _ndarray_wrapper

        submodule_dict = {}
        for op_name_prefix in base._OP_NAME_PREFIX_LIST:
            submodule_dict[op_name_prefix] =\
                    getattr(module, op_name_prefix[1:-1])

        wrap_list = target_precision_ops if target_precision_ops is not None \
                        else lists.symbol.FP16_FUNCS
        for fun_name in wrap_list:
            try:
                fun_name, cur_module = self._get_fun_to_wrap(fun_name, module, submodule_dict)
                f_to_wrap = getattr(cur_module, fun_name)
                setattr(cur_module, fun_name, _wrapper(f_to_wrap))
                if cur_module == module:
                    setattr(module.op, fun_name, _wrapper(f_to_wrap))
            except AttributeError:
                pass

        wrap_list = fp32_ops if fp32_ops is not None else lists.symbol.FP32_FUNCS
        for fun_name in wrap_list:
            try:
                fun_name, cur_module = self._get_fun_to_wrap(fun_name, module, submodule_dict)
                f_to_wrap = getattr(cur_module, fun_name)
                setattr(cur_module, fun_name, _wrapper(f_to_wrap))
                if cur_module == module:
                    setattr(module.op, fun_name, _wrapper(f_to_wrap))
            except AttributeError:
                pass

        #! TODO(huhanpeng) do not apply sleep functions for conditional_fp32_ops, e.g., 'Activation', 'act_type', ['softrelu']),
        # wrap_list = conditional_fp32_ops if conditional_fp32_ops is not None \
        #                 else lists.symbol.CONDITIONAL_FP32_FUNCS
        # for fun_name, arg, arg_values in wrap_list:
        #     try:
        #         fun_name, cur_module = self._get_fun_to_wrap(fun_name, module, submodule_dict)
        #         f_to_wrap = getattr(cur_module, fun_name)
        #         setattr(cur_module, fun_name, _wrapper(f_to_wrap))
        #         if cur_module == module:
        #             setattr(module.op, fun_name, _wrapper(f_to_wrap))
        #     except AttributeError:
        #         pass

        #! TODO(huhanpeng) do not apply sleep functions for cast, e.g., broadcast... div, _plus_scalar...
        # for fun_name in lists.symbol.WIDEST_TYPE_CASTS:
        #     try:
        #         fun_name, cur_module = self._get_fun_to_wrap(fun_name, module, submodule_dict)
        #         f_to_wrap = getattr(cur_module, fun_name)
        #         setattr(cur_module, fun_name, _symbol_widest_wrapper(f_to_wrap))
        #         if cur_module == module:
        #             setattr(module.op, fun_name, _symbol_widest_wrapper(f_to_wrap))
        #     except AttributeError:
        #         pass

    def _wrap_loss_output_functions(self, module):
        if module == ndarray:
            def _wrapper(f):
                def _scaling_wrapper(*args, **kwargs):
                    time.sleep(self.SLEEP_TIME)
                    return f(*args, **kwargs)
                _scaling_wrapper.__name__ = f.__name__
                _scaling_wrapper.__module__ = f.__module__
                _scaling_wrapper.__doc__ = f.__doc__
                return _scaling_wrapper
        else:
            def _wrapper(f):
                def _warning_wrapper(*args, **kwargs):
                    logging.warning("%s does not support dynamic loss scaling "
                                    "in symbolic and hybridized execution.", f.__name__)
                    time.sleep(self.SLEEP_TIME)
                    return f(*args, **kwargs)
                _warning_wrapper.__name__ = f.__name__
                _warning_wrapper.__module__ = f.__module__
                _warning_wrapper.__doc__ = f.__doc__
                return _warning_wrapper

        for fun_name in lists.symbol.LOSS_OUTPUT_FUNCTIONS:
            try:
                f_to_wrap = getattr(module, fun_name)
                setattr(module, fun_name, _wrapper(f_to_wrap))
            except AttributeError:
                pass



