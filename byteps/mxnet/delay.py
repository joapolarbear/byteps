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

import ctypes
import time
import os
import sysconfig
def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'

dll_path = os.path.join(os.path.dirname(__file__),
                        'c_lib' + get_ext_suffix())
MXNET_LIB_CTYPES = ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL)

class Delayer:
    def __init__(self):
        _delay = os.getenv("BYTEPS_TRACE_DELAY_CMP", None)
        if _delay is None:
            return
        self.SLEEP_TIME = int(_delay * 1)
        self._initialized = False
        if not self._initialized:
            self._initialized = True
            logging.info("Using synthetic delays.")
            self._wrap_symbol_functions(symbol)
            self._wrap_symbol_functions(ndarray)
            self._wrap_loss_output_functions(ndarray)
            self._wrap_loss_output_functions(symbol)

    def _do_sleep(self, f, name="symbol"):
        _s = time.time()
        MXNET_LIB_CTYPES.byteps_mxnet_sleep(ctypes.c_int(self.SLEEP_TIME), ctypes.c_bool(1))
        # time.sleep(self.SLEEP_TIME / 1000.0)
        # print("%s: %s sleep for %f s" % (name, f.__name__, time.time() - _s))

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
                self._do_sleep(f, name="ndarray")
                return f(*args, **kwargs)
            _new_fun.__name__ = f.__name__
            _new_fun.__module__ = f.__module__
            _new_fun.__doc__ = f.__doc__
            return _new_fun

        def _symbol_wrapper(f):
            def _new_fun(*args, **kwargs):
                self._do_sleep(f)
                return f(*args, **kwargs)
            _new_fun.__name__ = f.__name__
            _new_fun.__module__ = f.__module__
            _new_fun.__doc__ = f.__doc__
            return _new_fun

        def _symbol_widest_wrapper(f):
            def _new_fun(*args, **kwargs):
                self._do_sleep(f)
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

        for fun_name in lists.symbol.FP16_FUNCS:
            try:
                fun_name, cur_module = self._get_fun_to_wrap(fun_name, module, submodule_dict)
                f_to_wrap = getattr(cur_module, fun_name)
                setattr(cur_module, fun_name, _wrapper(f_to_wrap))
                if cur_module == module:
                    setattr(module.op, fun_name, _wrapper(f_to_wrap))
            except AttributeError:
                pass

        for fun_name in lists.symbol.FP32_FUNCS:
            try:
                fun_name, cur_module = self._get_fun_to_wrap(fun_name, module, submodule_dict)
                f_to_wrap = getattr(cur_module, fun_name)
                setattr(cur_module, fun_name, _wrapper(f_to_wrap))
                if cur_module == module:
                    setattr(module.op, fun_name, _wrapper(f_to_wrap))
            except AttributeError:
                pass

        #! TODO(huhanpeng) do not apply sleep functions for conditional_fp32_ops, e.g., 'Activation', 'act_type', ['softrelu']),
        # for fun_name, arg, arg_values in lists.symbol.CONDITIONAL_FP32_FUNCS:
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
                    self._do_sleep(f, name="ndarray")
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
                    self._do_sleep(f)
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



