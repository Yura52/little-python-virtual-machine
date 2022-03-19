#!/usr/bin/env python3


# TODO: uncomment matmul.
# TODO: usage of top_data is not intuitive.
# TODO: default positional and keyword-only arguments in LittleFunctions
#       are kept in a different style. Fix it.
# TODO: implement almost useless error-hierarchy
# TODO: add comments.


import sys
import dis
import inspect
import operator
import types
import time


# === general staff ==========================================================
def print_err(*args, **kwargs):
    """Print whatever to sys.stderr."""
    print(file=sys.stderr, *args, **kwargs)


def dump_vars(var_dict, only_user_vars=True):
    """Dump variables from a dict.

    Parameters
    ----------
    var_dict : dict
        Contains names of variables and their values.

    only_user_vars : bool
        If True, the function will try to avoid printing info
        about modules and special variables.
    """
    for name in var_dict:
        if (only_user_vars and
                (isinstance(var_dict[name], types.ModuleType) or
                    name.startswith('_'))):
            continue
        value = var_dict[name]
        print_err('{} {} [{}]'.format(name, value, id(value)))


def getstdstreams():
    """Return current standard streams.

    Returns
    -------
    sys.stdin

    sys.stdout

    sys.stderr
    """
    return sys.stdin, sys.stdout, sys.stderr


def setstdstreams(stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr):
    """Set standard streams.

    Parameters
    ----------
    stdin : stream-like
        Value for sys.stdin.

    stdout : stream-like
        Value for sys.stdout.

    stderr : stream-like
        Value for sys.stderr.
    """
    sys.stdin = stdin
    sys.stdout = stdout
    sys.stderr = stderr


# === LittleError ============================================================
class LittleError(Exception):
    """"""
    pass


class LittleBytecodeError(LittleError):
    """"""
    pass


class LittleTechnicalError(LittleError):
    """"""
    pass


class LittleStackError(LittleTechnicalError):
    """"""
    pass


class LittleRuntimeError(LittleError):
    """"""
    pass


class LittleNameError(LittleRuntimeError):
    """"""
    pass


# === LittleBlock ============================================================
class LittleBlock(object):
    """Auxiliary class for execution control.

    Is used when handling loops, try-except constructions etc.
    """

    def __init__(self, type_, handler, level):
        """Initialize a LittleBlock object.

        Parameters
        ----------
        type_ : string
            Must be one of {'loop'}.

        handler : int
            Number of an instraction, where to jump to find a handler.

        level : int
            Size which data_stack which must be equal to when
            the block is just popped.
        """
        self.b_type = type_
        self.b_handler = handler
        self.b_level = level


# === LittleCell =============================================================
class LittleCell(object):
    """Analogue of standard cell.

    Is used when dealing with closures.
    """

    def __init__(self, value):
        """Initialize a LittleCell object.

        Parameters
        ----------
        value : any type
            Value to keep.
        """
        self.value = value

    def get(self):
        """Return value which the cell contains."""
        return self.value

    def set(self, new_value):
        """Set value which the cell contains."""
        self.value = new_value


# === LittleFunction =========================================================
class LittleFunction(object):
    """Class for creating and calling functions.

    1.  A noticable advantage of LittleFunction is the ability to get
        information on all default arguments by using the attribute
        'func_all_defaults'

    2.  Objects of the class are not callable.
        It is important: making them callable imply having reference to
        an object of the class LittleVirtualMachine and ability to induce
        some changes in it that is not logically correct.
    """

    namespace_attrs = ['func_globals', 'func_all_defaults', 'func_closure']
    nonverbose_dump_attrs = ['func_doc']
    verbose_dump_attrs = [
        'func_doc',
        'func_qualname',
        'func_pos_names',
        'func_pos_defaults',
        'func_kw_defaults',
    ]

    def __init__(self, code, globals_, defaults, kwdefaults,
                 qualname, closure=()):
        """Initialize a LittleFunction object.

        Parameters
        ----------
        code : code_object
            Code object of the function.

        globals_ : dict
            Global namespace for the function calls.

        defaults : array-like
            Default positional arguments.

        kwdefaults : dict
            Default keyword-only arguments.

        qualname : string
            Full name of the function.

        closure : array-like
            Cells for function-closures.
        """
        self.func_code = code
        self.func_globals = globals_
        self.func_pos_defaults = tuple(defaults)
        self.func_kw_defaults = kwdefaults
        self.func_qualname = qualname

        self.func_closure = {}
        for name, cell in zip(code.co_freevars, closure):
            self.func_closure[name] = cell

        var_names = code.co_varnames
        pos_count = code.co_argcount
        self.func_pos_names = tuple(var_names[:pos_count])
        self.func_kwonly_names = var_names[pos_count:self.func_unpacked_count]

        self.func_all_defaults = {}
        for name, value in zip(reversed(self.func_pos_names),
                               reversed(tuple(self.func_pos_defaults))):
            self.func_all_defaults[name] = value
        self.func_all_defaults.update(self.func_kw_defaults)

    @property
    def func_name(self):
        return self.func_code.co_name

    @property
    def func_doc(self):
        return self.func_code.__doc__

    @property
    def func_unpacked_count(self):
        return self.func_code.co_argcount + self.func_code.co_kwonlyargcount

    @property
    def func_varnames(self):
        return self.func_code.co_varnames

    def dump(self, only_user_vars=True, verbose=False):
        """"""
        print_err('===== LittleFunction[{}] ====='.format(id(self)))
        print_err('func_name', getattr(self, 'func_name'))

        for namespace in LittleFunction.namespace_attrs:
            print_err('----------')
            print_err(namespace)
            dump_vars(getattr(self, namespace), only_user_vars)

        if not verbose:
            attrs_to_dump = LittleFunction.nonverbose_dump_attrs
        else:
            attrs_to_dump = LittleFunction.verbose_dump_attrs
        for attr in attrs_to_dump:
            print_err('--------\n')
            print_err(attr, getattr(self, attr))
        print_err('========================================')


# === LittleFrame ============================================================
class LittleFrame(object):
    """Class for implementing context.

    An analogue of types.FrameObject, which is used for function calls and
    everything that implies change of the context.

    Has two stacks: for keeping data and LittleBlock objects.
    """

    finish_states = ['returned']
    namespace_attrs = ['f_globals', 'f_locals', 'f_cells']
    nonverbose_dump_attrs = ['f_datastack']
    verbose_dump_attrs = [
        'f_datastack',
        'f_state',
        'f_lasti',
        'f_back',
        'f_blockstack',
        'f_first_unused_const_index'
    ]

    def __init__(self, f_code, f_globals, f_locals, f_cells, f_back):
        """Initialize an LittleFrame object.

        Parameters
        ----------
        f_code : code_object
            Code object to run.

        f_globals : dict
            Global namespace of the execution context.

        f_locals : dict
            Local namespace of the execution context.

        f_cells : dict
            Mapping 'name of free variable' to 'cell'.

        f_back : LittleFrame
            The frame from which the command for creating a new one came.
        """
        self.f_code = f_code
        self.f_lasti = 0  # index of the next-to-process byte-command
        self.f_back = f_back

        self.f_globals = f_globals
        self.f_locals = f_locals
        if f_back is not None:
            self.f_builtins = f_back.f_builtins
        else:
            self.f_builtins = f_globals['__builtins__']

        self.f_cells = f_cells

        self.f_datastack = []
        self.f_blockstack = []

        self.f_state = 'just_built'

        # It is a hint for the method 'op_load_closure'.
        # The idea is to use the first unused constant from the code object
        # in the case of absense the goal-object among cells or local
        # variable.
        # See the method 'op_load_closure' for better understanding.
        self.f_first_unused_const_index = 0

    def is_terminated(self):
        """"""
        return self.f_state in LittleFrame.finish_states

    def dump(self, only_user_vars=True, verbose=False):
        """"""
        print_err('===== LittleFrame[{}] ====='.format(id(self)))
        for namespace in LittleFrame.namespace_attrs:
            print_err('----------')
            print_err(namespace)
            dump_vars(getattr(self, namespace), only_user_vars)

        if not verbose:
            attrs_to_dump = LittleFrame.nonverbose_dump_attrs
        else:
            attrs_to_dump = LittleFrame.verbose_dump_attrs
        for attr in attrs_to_dump:
            print_err('--------\n')
            print_err(attr, getattr(self, attr))
        print_err('========================================')


# === LittleVirtualMachine ===================================================
class LittleVirtualMachine(object):
    """A little python 3 bytecode interpreter.

    1.  See the method 'run_code' to learn how to have fun.

    2.  Although little, has wild range of debug settings.
        See attribute 'debug_settings' and method 'debug_job'.

    3.  If any command is supported, you can be sure that it is
        FULLY supported (well, I hope so).
        Examples:
            -   functions can be used with any combinations of
                arguments (positional, kw-only, default) and any
                ways of passing arguments can be used;
            -   closures allow to use decorators with arguments
            -   any level of nesting functions, comprehensions and loops
                is supported
    """

    unary_ops = {
        'positive': operator.pos,
        'negative': operator.neg,
        'not': operator.not_,
        'invert': operator.inv
    }

    binary_ops = {
        'power': operator.pow,
        'multiply': operator.mul,
        'modulo': operator.mod,
        'add': operator.add,
        'subtract': operator.sub,
        'subscr': operator.getitem,
        'floor_divide': operator.floordiv,
        'true_divide': operator.truediv,
        'lshift': operator.lshift,
        'rshift': operator.rshift,
        'and': operator.and_,
        'xor': operator.xor,
        'or': operator.or_,
        # new in python 3.5
        # 'matmul': operator.matmul
    }

    inplace_ops = {
        'floor_divide': operator.ifloordiv,
        'true_divide': operator.itruediv,
        'modulo': operator.imod,
        'add': operator.iadd,
        'subtract': operator.isub,
        'multiply': operator.imul,
        'power': operator.ipow,
        'lshift': operator.ilshift,
        'rshift': operator.irshift,
        'and': operator.iand,
        'xor': operator.ixor,
        'or': operator.ior,
        # new in python 3.5
        # 'matmul': operator.imatmul
    }

    compare_ops = {
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        '>=': operator.ge,
        '>': operator.gt,
        'in': operator.contains,
        'not in': lambda x, y: not operator.contains(x, y),
        'is': operator.is_,
        'is not': operator.is_not
    }

    def __init__(self):
        """Initialize a LittleVirtualMachine object."""
        self.frame_stack = []
        self.current_frame = None

        # some of fields relate to the LittleVirtualMachine object itself
        # some of fields activate/deactivate debug output for particular
        # methods
        self.debug_settings = {
            'debug_mode': False,
            'dump_dis': True,
            'time_between_ops': 0.25,
            'only_user_vars': True,
            'trace_jumps': False,
            'output_separator': '-----------------------',

            'run_next_op': True,
            'op_call_function': True,
            'call_little_function': False,

            'verbose': False
        }

# === work with LittleBlock ==================================================
    def push_block(self, type_, handler, level=None):
        """"""
        if level is None:
            level = self.current_data_stack_size()
        block = LittleBlock(type_, handler, level)
        self.current_frame.f_blockstack.append(block)

    def pop_block(self):
        """"""
        self.current_frame.f_blockstack.pop()

    def get_block_attr(self, pos, attr):
        """"""
        return getattr(self.current_frame.f_blockstack[pos], attr)

    def get_top_block_attr(self, attr):
        """"""
        return self.get_block_attr(-1, attr)

    def clear_until_level(self, level):
        """"""
        self.pop_data(self.current_data_stack_size() - level)

# === work with LittleFunction ===============================================
    def call_little_function(self, func, *args, **kwargs):
        """"""
        pos_from_kwargs = {}
        pos_names = list(func.func_pos_names).copy()

        for name in pos_names:
            if name in kwargs:
                pos_from_kwargs[name] = kwargs[name]
        for name in pos_from_kwargs:
            del kwargs[name]
            pos_names.remove(name)

        args = list(args)
        pos = {}
        for name, value in zip(pos_names, args):
            pos[name] = value
        for name in pos:
            args.remove(pos[name])
            pos_names.remove(name)
        pos.update(pos_from_kwargs)

        kwonly = {}
        kwonly_names = list(func.func_kwonly_names)
        for name in kwonly_names:
            if name in kwargs:
                kwonly[name] = kwargs[name]
        for name in kwonly:
            del kwargs[name]
            kwonly_names.remove(name)
        # now all pos and kwonly arguments got their values

        var_names = func.func_varnames
        fisrt_unused_arg_index = func.func_unpacked_count
        varargs = {}
        if func.func_code.co_flags & inspect.mod_dict['CO_VARARGS']:
            varargs[var_names[fisrt_unused_arg_index]] = tuple(args)
            fisrt_unused_arg_index += 1

        callargs = func.func_all_defaults.copy()
        for args_ in [pos, kwonly, varargs]:
            callargs.update(args_)
        if func.func_code.co_flags & inspect.mod_dict['CO_VARKEYWORDS']:
            callargs[var_names[fisrt_unused_arg_index]] = kwargs

        self.debug_job('call_little_function', func, callargs)
        frame = self.build_frame(func.func_code, callargs, func.func_globals,
                                 f_cells=func.func_closure)
        return self.run_frame(frame)

# === work with LittleFrame ==================================================
    def build_frame(self, f_code, callargs={}, f_globals=None, f_locals=None,
                    f_cells={}):
        """"""
        if f_globals is None:
            if self.current_frame is None:
                f_globals = {
                    '__builtins__': __builtins__,
                    '__name__': '__main__',
                    '__spec__': None,
                    '__doc__': None,
                    '__package__': None
                }
                if hasattr(__builtins__, '__dict__'):
                    f_globals['__builtins__'] = __builtins__.__dict__
                f_locals = f_globals
            else:
                f_globals = self.current_frame.f_globals
                f_locals = {}
        if f_locals is None:
            f_locals = f_globals.copy()
        f_locals.update(callargs)

        return LittleFrame(f_code, f_globals, f_locals, f_cells,
                           self.current_frame)

    def push_frame(self, frame):
        """"""
        self.frame_stack.append(frame)
        self.current_frame = frame

    def pop_frame(self):
        """"""
        self.frame_stack.pop()
        if self.frame_stack:
            self.current_frame = self.frame_stack[-1]
        else:
            self.current_frame = None

    def set_frame_attr(self, pos, attr, value):
        """"""
        setattr(self.frame_stack[pos], attr, value)

    def get_frame_attr(self, pos, attr):
        """"""
        return getattr(self.frame_stack[pos], attr)

    def set_current_frame_attr(self, attr, value):
        """"""
        self.set_frame_attr(-1, attr, value)

    def get_current_frame_attr(self, attr):
        """"""
        return self.get_frame_attr(-1, attr)

    def lookup_name(self, name, local=True):
        """"""
        if local:
            for frame in self.frame_stack[::-1]:
                if name in frame.f_locals:
                    return frame.f_locals[name]
        if name in self.current_frame.f_globals:
            return self.current_frame.f_globals[name]
        if name in self.current_frame.f_builtins:
            return self.current_frame.f_builtins[name]
        raise LittleNameError(name)

    def run_frame(self, frame):
        """"""
        self.push_frame(frame)
        self.set_current_frame_attr('f_state', 'running')
        while not self.current_frame.is_terminated():
            self.run_next_op()
        ret = self.get_current_frame_attr('f_result')
        self.pop_frame()
        return ret

# === work with LittleFrame.data_stack =======================================
    def push_data(self, *new_data):
        """"""
        self.current_frame.f_datastack.extend(new_data)

    def pop_data(self, pop_count=None):
        """"""
        if pop_count is None:
            return self.current_frame.f_datastack.pop()
        if pop_count == 0:
            return []
        if pop_count > self.current_data_stack_size():
            report = 'too little stack to pop {} values'.format(pop_count)
            raise LittleStackError(report)
        ret = self.current_frame.f_datastack[-pop_count:]
        del self.current_frame.f_datastack[-pop_count:]
        return ret

    def top_data(self, pos=0):
        """"""
        if pos > (self.current_data_stack_size() - 1):
            report = 'too little stack to see top {}'.format(pos)
            raise LittleStackError(report)
        return self.current_frame.f_datastack[-pos - 1]

    def current_data_stack_size(self):
        """"""
        return len(self.current_frame.f_datastack)

# === work with ops ==========================================================
    def get_op_code(self):
        """"""
        op_index = self.current_frame.f_lasti
        op_code = self.current_frame.f_code.co_code[op_index]
        self.current_frame.f_lasti += 1
        return op_code

    def get_op_name(self, op_code):
        """"""
        try:
            return dis.opname[op_code].lower()
        except KeyError:
            raise LittleBytecodeError('unknown byte-command' + str(op_code))

    def get_op_arg(self, op_code):
        """"""
        if op_code < dis.HAVE_ARGUMENT:
            return None

        f_code = self.current_frame.f_code
        f_lasti = self.current_frame.f_lasti
        op_arg = f_code.co_code[f_lasti] + (f_code.co_code[f_lasti + 1] << 8)
        self.current_frame.f_lasti += 2

        if op_code in dis.hasconst:
            if op_arg > self.current_frame.f_first_unused_const_index:
                self.current_frame.f_first_unused_const_index += 1
            op_arg = f_code.co_consts[op_arg]
        elif op_code in dis.hasname:
            op_arg = f_code.co_names[op_arg]
        elif op_code in dis.haslocal:
            op_arg = f_code.co_varnames[op_arg]
        elif op_code in dis.hascompare:
            op_arg = dis.cmp_op[op_arg]
        elif op_code in dis.hasfree:
            cellvars = self.get_current_frame_attr('f_code').co_cellvars
            cellvars_len = len(cellvars)
            if op_arg < cellvars_len:
                op_arg = cellvars[op_arg]
            else:
                freevars = self.get_current_frame_attr('f_code').co_freevars
                op_arg = freevars[op_arg - cellvars_len]

        return op_arg

    def run_next_op(self):
        """"""
        op_code = self.get_op_code()
        op_name = self.get_op_name(op_code)
        op_arg = self.get_op_arg(op_code)
        self.debug_job('run_next_op', op_name, op_arg)

        if (op_name.startswith('call_function')):
            self.op_call_function(op_name, op_arg)
            return

        if (op_name.startswith('unary') or
                op_name.startswith('binary') or
                op_name.startswith('inplace')):
            getattr(self, 'op_' + op_name.split(sep='_')[0])(op_name)
            return

        if op_code < dis.HAVE_ARGUMENT:
            getattr(self, 'op_' + op_name)()
        else:
            getattr(self, 'op_' + op_name)(op_arg)

# === work with code_object ==================================================
    def run_code(self, code_obj,
                 globals_=None, locals_=None,
                 stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr,
                 debug_settings={}):
        """"""
        self.manage_debug_settings(debug_settings, code_obj)

        old_std_streams = getstdstreams()
        setstdstreams(stdin, stdout, stderr)

        start_frame = self.build_frame(code_obj,
                                       f_globals=globals_,
                                       f_locals=locals_)
        self.run_frame(start_frame)

        setstdstreams(*old_std_streams)

# === basic ops===============================================================
    def op_nop(self):
        """"""
        pass

    def op_pop_top(self):
        """"""
        self.pop_data()

    def op_rot_two(self):
        """"""
        self.push_data(*reversed(self.pop_data(2)))

    def op_rot_three(self):
        """"""
        tos_2, tos_1, tos = self.pop_data(3)
        self.push_data(tos, tos_2, tos_1)

    def op_dup_top(self):
        self.push_data(self.top_data())

    def op_dup_top_two(self):
        self.push_data(self.top_data(1))
        self.push_data(self.top_data(1))

    def op_return_value(self):
        """"""
        self.set_current_frame_attr('f_result', self.pop_data())
        self.set_current_frame_attr('f_state', 'returned')

# === operators ==============================================================
    def op_unary(self, op_name):
        """"""
        op = self.unary_ops[op_name[len('unary_'):]]
        self.push_data(op(self.pop_data()))

    def op_binary(self, op_name):
        """"""
        op = self.binary_ops[op_name[len('binary_'):]]
        self.push_data(op(*self.pop_data(2)))

    def op_inplace(self, op_name):
        """"""
        op = self.inplace_ops[op_name[len('inplace_'):]]
        self.current_frame.f_datastack[-1] =\
            op(self.current_frame.f_datastack[-2], self.pop_data())

    def op_compare_op(self, op_name):
        """"""
        op = self.compare_ops[op_name]
        if op_name not in ['in', 'not in']:
            self.push_data(op(*self.pop_data(2)))
        else:
            self.push_data(op(*reversed(self.pop_data(2))))

# === built & unpack =========================================================
    def op_build_tuple(self, item_count):
        """"""
        self.push_data(tuple(self.pop_data(item_count)))

    def op_build_list(self, item_count):
        """"""
        self.push_data(list(self.pop_data(item_count)))

    def op_build_set(self, item_count):
        """"""
        self.push_data(set(self.pop_data(item_count)))

    def op_build_map(self, item_count):
        """"""
        if sys.version_info.minor >= 5:
            it = iter(self.pop_data(2 * item_count))
            dict_ = {}
            for key, value in zip(it, it):
                dict_[key] = value
            self.push_data(dict_)
        else:
            self.push_data({})

    # deprecated in python 3.5
    def op_store_map(self):
        """"""
        value, key = self.pop_data(2)
        dict.__setitem__(self.top_data(), key, value)

    def op_list_append(self, index):
        """"""
        list.append(self.top_data(index), self.pop_data())

    def op_set_add(self, index):
        """"""
        set.add(self.top_data(index), self.pop_data())

    def op_map_add(self, index):
        """"""
        value, key = self.pop_data(2)
        dict.__setitem__(self.top_data(index - 1), key, value)

    def op_build_slice(self, argc):
        """"""
        self.push_data(slice(*self.pop_data(argc)))

    def op_unpack_sequence(self, count):
        """"""
        self.push_data(*reversed(self.pop_data()))

    def op_unpack_ex(self, arg):
        """"""
        before_list_count = arg & 0xFF
        after_list_count = arg >> 8
        values_to_assign = list(self.pop_data())

        if after_list_count > 0:
            self.push_data(*reversed(values_to_assign[-after_list_count:]))
            sss = slice(before_list_count, -after_list_count)
            self.push_data(values_to_assign[sss])
        else:
            self.push_data(values_to_assign[before_list_count:])
        if before_list_count > 0:
            self.push_data(*reversed(values_to_assign[:before_list_count]))

# === load & store ===========================================================
    def op_store_name(self, name):
        """"""
        self.current_frame.f_locals[name] = self.pop_data()

    def op_store_fast(self, name):
        """"""
        self.current_frame.f_locals[name] = self.pop_data()

    def op_store_global(self, name):
        """"""
        self.current_frame.f_globals[name] = self.pop_data()

    def op_store_subscr(self):
        """"""
        value, obj, index = self.pop_data(3)
        operator.setitem(obj, index, value)

    def op_load_const(self, value):
        """"""
        self.push_data(value)

    def op_load_name(self, name):
        """"""
        self.push_data(self.lookup_name(name))

    def op_load_fast(self, name):
        """"""
        self.push_data(self.lookup_name(name))

    def op_load_global(self, name):
        """"""
        self.push_data(self.lookup_name(name, local=False))

    def op_delete_name(self, name):
        """"""
        del self.current_frame.f_locals[name]

    def op_delete_fast(self, name):
        """"""
        del self.current_frame.f_locals[name]

    def op_delete_global(self, name):
        """"""
        del self.current_frame.f_globals[name]

    def op_delete_subscr(self):
        """"""
        operator.delitem(*self.pop_data(2))

# === functions ==============================================================
    def op_call_function(self, op_name, argc):
        """"""
        packed_kwargs = {}
        if 'kw' in op_name:
            packed_kwargs = self.pop_data()
        packed_posargs = []

        kwargs = {}
        kwarg_count = argc >> 8
        it = iter(self.pop_data(2 * kwarg_count))
        for key, value in zip(it, it):
            dict.__setitem__(kwargs, key, value)
        kwargs.update(packed_kwargs)

        if 'var' in op_name:
            packed_posargs = list(self.pop_data())
        posarg_count = argc & 0xFF
        posargs = self.pop_data(posarg_count) + packed_posargs

        func = self.pop_data()
        self.debug_job('op_call_function', func, posargs, kwargs)

        if isinstance(func, LittleFunction):
            self.push_data(self.call_little_function(func,
                                                     *posargs,
                                                     **kwargs))
        elif not hasattr(func, '__name__'):
            self.push_data(func(*posargs, **kwargs))
        else:
            if getattr(func, '__name__') != '__build_class__':
                self.push_data(func(*posargs, **kwargs))
            else:
                little_func = posargs[0]
                crutch = types.FunctionType(
                    little_func.func_code,
                    little_func.func_globals,
                    argdefs=little_func.func_pos_defaults
                )
                posargs[0] = crutch
                self.push_data(func(*tuple(posargs), **kwargs))

    def op_make_function(self, argc):
        """"""
        code, qualname = self.pop_data(2)

        kwdefault_count = argc >> 8
        it = iter(self.pop_data(2 * kwdefault_count))
        kwdefaults = {}
        for key, value in zip(it, it):
            kwdefaults[key] = value

        default_count = argc & 0xFF
        defaults = self.pop_data(default_count)

        self.push_data(LittleFunction(
                       code,
                       self.get_current_frame_attr('f_globals'),
                       defaults,
                       kwdefaults,
                       qualname))

# === closures ===============================================================
    def op_make_closure(self, argc):
        """"""
        closure, code, qualname = self.pop_data(3)

        kwdefault_count = argc >> 8
        it = iter(self.pop_data(2 * kwdefault_count))
        kwdefaults = {}
        for key, value in zip(it, it):
            kwdefaults[key] = value

        default_count = argc & 0xFF
        defaults = self.pop_data(default_count)

        self.push_data(LittleFunction(
                       code,
                       self.get_current_frame_attr('f_globals'),
                       defaults,
                       kwdefaults,
                       qualname,
                       closure))

    def op_load_closure(self, name):
        """"""
        f_cells = self.get_current_frame_attr('f_cells')
        f_locals = self.get_current_frame_attr('f_locals')
        # consts = self.get_current_frame_attr('f_code').co_consts

        if name in f_cells:
            self.push_data(f_cells[name])
        elif name in f_locals:
            f_cells[name] = LittleCell(f_locals[name])
            self.push_data(f_cells[name])
        else:
            ind = self.current_frame.f_first_unused_const_index + 1
            const = self.current_frame.f_code.co_consts[ind]
            f_cells[name] = LittleCell(const)
            self.push_data(f_cells[name])

    def op_load_deref(self, name):
        """"""
        f_cells = self.get_current_frame_attr('f_cells')
        self.push_data(f_cells[name].get())

    def op_load_classderef(self, name):
        """"""
        f_locals = self.get_current_frame_attr('f_locals')
        if name in f_locals:
            self.push_data(f_locals[name])
        else:
            self.push_data(self.get_current_frame_attr('f_cells')[name].get())

    def op_store_deref(self, name):
        """"""
        f_cells = self.get_current_frame_attr('f_cells')
        if name in f_cells:
            f_cells[name].set(self.pop_data())
        else:
            f_cells[name] = LittleCell(self.pop_data())

    def op_delete_deref(self, arg):
        """"""
        del self.get_current_frame_attr('f_cells')[name]

# === generators =============================================================
    def op_yield_value(self):
        """The reason why 101/102."""
        pass

# === loops ==================================================================
    def op_get_iter(self):
        """"""
        self.push_data(iter(self.pop_data()))

    def op_get_yield_from_iter(self):
        """"""
        tos = self.top_data()
        if not isinstance(tos, types.GeneratorType)\
                and not isinstance(tos, types.coroutine):
            self.push_data(iter(self.pop_data()))

    def op_for_iter(self, op_offset):
        """"""
        try:
            self.push_data(next(self.top_data()))
        except StopIteration:
            self.pop_data()
            current_lasti = self.get_current_frame_attr('f_lasti')
            self.set_current_frame_attr('f_lasti', current_lasti + op_offset)

    def op_setup_loop(self, offset):
        """"""
        current_lasti = self.get_current_frame_attr('f_lasti')
        self.push_block('loop', current_lasti + offset - 1)

    def op_break_loop(self):
        """"""
        handler_index = self.get_top_block_attr('b_handler')
        level = self.get_top_block_attr('b_level')
        self.clear_until_level(level)
        self.set_current_frame_attr('f_lasti', handler_index)

    def op_continue_loop(self, op_index):
        """"""
        self.set_current_frame_attr('f_lasti', op_index)

# === execution control ======================================================
    def op_jump_absolute(self, op_index):
        """"""
        self.set_current_frame_attr('f_lasti', op_index)

    def op_jump_forward(self, op_offset):
        """"""
        current_lasti = self.get_current_frame_attr('f_lasti')
        self.set_current_frame_attr('f_lasti', current_lasti + op_offset)

    def op_pop_jump_if_true(self, op_index):
        """"""
        if self.pop_data():
            self.set_current_frame_attr('f_lasti', op_index)

    def op_pop_jump_if_false(self, op_index):
        """"""
        if not self.pop_data():
            self.set_current_frame_attr('f_lasti', op_index)

    def op_jump_if_true_or_pop(self, op_index):
        """"""
        if self.top_data():
            self.set_current_frame_attr('f_lasti', op_index)
        else:
            self.pop_data()

    def op_jump_if_false_or_pop(self, op_index):
        """"""
        if not self.top_data():
            self.set_current_frame_attr('f_lasti', op_index)
        else:
            self.pop_data()

    def op_pop_block(self):
        """"""
        self.pop_block()

# === classes ================================================================
    def op_store_attr(self, attr):
        """"""
        obj, value = reversed(self.pop_data(2))
        setattr(obj, attr, value)

    def op_delete_attr(self, attr):
        """"""
        delattr(self.pop_data(), attr)

    def op_load_attr(self, attr):
        """"""
        self.push_data(getattr(self.pop_data(), attr))

    def op_load_build_class(self):
        """"""
        self.push_data(__build_class__)

# === debug ==================================================================
    @property
    def debug_mode(self):
        """"""
        return self.debug_settings['debug_mode']

    def manage_debug_settings(self, debug_settings, code_obj):
        """"""
        self.debug_settings.update(debug_settings)
        if not self.debug_mode:
            return

        self.debug_handlers = {
            'op_call_function': self.debug_op_call_function,
            'run_next_op': self.debug_run_next_op,
            'call_little_function': self.debug_call_little_function,
        }

        debug_settings = self.debug_settings
        if debug_settings['trace_jumps']:
            sys.settrace(self.jump_tracer)
        if debug_settings['dump_dis']:
            print_err(dis.dis(code_obj), '\n')
        if debug_settings['run_next_op']:
            header = 'f_lasti' + (' ' * 3)\
                     + 'op_name' + (' ' * 18)\
                     + 'op_arg'
            print_err(header)
            print_err('{:-<41}'.format(''))

    def print_debug_sep(self):
        """"""
        print_err(self.debug_settings['output_separator'])

    def debug_job(self, caller, *args, **kwargs):
        """Does some debug-related job.

        If debug mode is off, does nothing.
        If debug for the caller is off, does nothing.
        In a different way, prints some useful information
        given by the coller.
        """
        if not self.debug_mode or not self.debug_settings[caller]:
            return
        self.debug_handlers[caller](*args, **kwargs)

    def debug_run_next_op(self, op_name, op_arg):
        """"""
        lasti = self.get_current_frame_attr('f_lasti')
        if dis.opmap[op_name.upper()] < dis.HAVE_ARGUMENT:
            lasti -= 1
        else:
            lasti -= 3
        report = '{:<10}{:<25}'.format(lasti, op_name)
        if op_arg is None:
            print_err(report)
        else:
            print_err(report + str(op_arg))
        time.sleep(self.debug_settings['time_between_ops'])

    def debug_op_call_function(self, func, posargs, kwargs):
        """"""
        self.print_debug_sep()
        print_err('function:', func)
        print_err('posargs:', posargs)
        print_err('kwargs:', kwargs)
        self.print_debug_sep()

    def debug_call_little_function(self, func, callargs):
        """"""
        self.print_debug_sep()
        only_user_vars = self.debug_settings['only_user_vars']
        verbose = self.debug_settings['verbose']
        func.dump(only_user_vars, verbose)
        print_err('callargs', callargs)
        self.print_debug_sep()

    def jump_tracer(self, frame, event, arg):
        """"""
        if event != 'call':
            return
        if frame.f_code.co_name in ['op_jump_forward',
                                    'op_jump_absolute',
                                    'op_jump_if_false_or_pop',
                                    'op_jump_if_true_or_pop',
                                    'op_pop_jump_if_false',
                                    'op_pop_jump_if_true',
                                    'op_break_loop',
                                    'op_continue_loop']:
            self.print_debug_sep()
            print_err('TO JUMP OR NOT TO JUMP?')
            self.print_debug_sep()


# === VirtualMachine =========================================================
class VirtualMachine(LittleVirtualMachine):
    """A wrapper for passing tests in Yandex School of Data Analysis."""
    pass


# === main ===================================================================
def main():
    """"""
    compiled = compile(sys.stdin.read(), '<stdin>', 'exec')
    # compiled = compile(open('test.py', 'r').read(), 'test.py', 'exec')

    debug_settings = {
        'debug_mode': True,
        'time_between_ops': 0.01,
        'call_little_function': True,
    }
    VirtualMachine().run_code(compiled)
    # VirtualMachine().run_code(compiled, debug_settings=debug_settings)


if __name__ == '__main__':
    main()
