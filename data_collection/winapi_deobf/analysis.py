import re
import networkx as nx
import random
import z3
import esil
import simplejson

class FunctionAnalysisError(Exception):pass

SUPPORTED_TYPES = [
    'mov', 'push', 'upush', 'pop', 'upop', 'xor', 'lea',
    'add', 'sub'
]

def is_supported(op):
    if op['type'] not in SUPPORTED_TYPES \
       or op['opcode'].startswith('rep ') \
       or op['family'] in ['priv', 'sse', 'mmx']:
        return False

    return True

def find_mem_address(memory, n):
    for addr, value in memory.iteritems():
        if value == n:
            addr = addr.split(':')[-1]
            if esil.INT_REGEX.match(addr):
                return int(addr)


def get_call_target(op, vm):
    if 'ptr' in op:
        return op['ptr']
    else:
        # check if it's a poiner to API passed via reg32
        ptr = op['opcode'].split(' ')[-1].strip()
        if ptr in vm.registers:
            val = vm.get_reg(ptr)
            if isinstance(val, z3.BitVecRef) and str(val).startswith('mem'):
                mem_addr = find_mem_address(vm.memory, val)
                
                if mem_addr:
                    return mem_addr

def get_iat_entry(target, import_table, functions):
    if target in import_table:
        # should be in 'ucall' case
        return import_table[target]
    else:
        # should be in some 'call' cases
        function = functions.get(target)
        if function and function.jmp_to_iat:
            return function.jmp_to_iat


def check_pointer(addr, exe):
    for sect in exe.get_sections():
        start = sect['vaddr']
        end = start  + sect['vsize']
        if addr >= start and addr < end:                
            return 'PTR'


def process_arg_str(s):
    # Check if it's arg or var
    name = esil.get_name_from_addr_str(s)
    if name:
        return name    
    # Fix negation
    # Examples: 4294967295*ret, 4294967295*ret
    s = re.sub(r'4294967295\*([a-zA-F0-9_]+)', lambda x: '-%s'%x.group(1), s)
    return s

        
def get_arguments_from_stack(n_args, vm, exe, max_args=12):
    args = []

    n_stack_val = len(vm.x86_stack)
        
    if n_stack_val > max_args:
        n_stack_val = max_args
        
    x = max([n_args, n_stack_val])
    for i in xrange(x):
        if vm.x86_stack:
            if i < n_args:
                val = vm.x86_stack.pop()
            else:
                val = vm.x86_stack[n_args - i - 1]
            arg = z3.simplify(val)
            
            if hasattr(arg, 'as_long'):
                ptr_ = check_pointer(arg.as_long(), exe)
                if ptr_:
                    arg = ptr_
                else:
                    arg = '0x%x' % arg.as_long()
            else:
                arg = str(arg)
                name = process_arg_str(arg)#
                if name: arg = name
        else:
            if i >= n_args:break
            arg = '*'
                
        args.append(arg)
    return args


def look_up_number_of_arguments(name):
    return 10
        

        

SYM_IMP_REGEX = re.compile(r'sym\.imp\.(?P<lib>.+?)_(?P<name>.+)')
IMPORT_REGEX = re.compile(r'(?P<lib>.+?\.(?:dll|drv|exe|sys|cpl))_(?P<name>.+)', re.I)

def get_import_table(r2):
    import_table = {}
    for imp in r2.get_import_table():
        ir = IMPORT_REGEX.match(imp['name'])
        import_table[imp.get('plt')] = {
            'lib': ir.group('lib'),
            'name': ir.group('name')
        }
    return import_table


def is_jmp_to_iat(ops, import_table):
    if len(ops) == 1:
        op = ops[0]
            
        if op.get('type') == 'jmp' and\
           op.get('ptr') in import_table:
                return import_table[op.get('ptr')]


def extract_functions(r2, import_table):
    r2_func_list = r2.get_functions()
    if not r2_func_list:
        return {}    

    functions = {}
    for i in xrange(len(r2_func_list)):
        r2_func = r2_func_list[i]
        if r2_func['offset'] in import_table:
            continue
        size = r2_func['size']
        # Let's skip ridiculous functions for now
        if size > 10000: continue

        ops = r2.get_disassembly(r2_func['offset'], size)

        if not ops: continue
        function_ = Function(r2_func['offset'], r2_func['name'], ops)
        function_.jmp_to_iat = is_jmp_to_iat(ops, import_table)
        functions[r2_func.get('offset')] = function_
        
    return functions

class WinAPILookup:
    def __init__(self):
        with open('winapi_deobf/winapi.json', 'rb') as fd:
            self.lookup = simplejson.load(fd)
        
    def look_up_number_of_arguments(self, api_name):
        # Mind A and W versions:
        res = self.lookup.get(api_name)
        if not res and (api_name.endswith('A') or api_name.endswith('W')):
            res = self.lookup.get(api_name[:-1])
            if not res:
                res = self.lookup.get(api_name.replace('Nt','Zw'))
                                        
        return res

class Function:
    def __init__(self, offset, name, ops):
        self.offset = offset
        self.name = name
        self.ops = ops
        self.end = ops[-1].get('offset')
        # Is it a Delphi-like call?
        self.jmp_to_iat = None


class CFG:
    def __init__(self, function_):
        self.ops = function_.ops
        # Instructions "mapped" in memory
        self.mem_map = {}
        self.offset = function_.offset
        self.end = function_.end
        self.cfg = nx.DiGraph()
        # Dict of basic blocks
        self.blocks = {}
        # Basic blocks that are returns or
        # tail calls
        self.exit_blocks = []
        self.n_edges = 0
        

    def __inside(self, addr):
        return addr >= self.offset and addr <= self.end
        
    def add_node(self, address):        
        if address not in self.cfg.nodes():
            # logging.debug('Adding node: 0x%x' % address)
            self.cfg.add_node(address)
            self.blocks[address] = []

    def add_edge(self, from_, to_):
        # logging.debug('Adding edge: 0x%x -> 0x%x' % (from_, to_))
        self.n_edges += 1
        self.cfg.add_edge(from_, to_)


    def build_graph(self):
        last_op_n = len(self.ops) - 1
        for i in xrange(len(self.ops)):
            op = self.ops[i]
            offset = op['offset']
            
            if i == last_op_n:
                op['next_addr'] = None
            else:
                op['next_addr'] = self.ops[i+1].get('offset')

            self.mem_map[offset] = op
            # Adding information about next address
            self.ops[i] = op

            # Identifying nodes
            if op['type'] in ['cjmp', 'jmp']:
                jmp_target = op.get('jump')
                if jmp_target: 
                    self.add_node(jmp_target)
                if 'fail' in op:
                    self.add_node(op['fail'])

        start_addr = self.offset

        self.__visited = []
        self.__reachable_blocks = set([start_addr])        

        self.add_node(start_addr)
        self.traverse(start_addr)

        # Check for tail calls
        if not self.exit_blocks:
            # Consider the last reachable block exit block
            last_block = sorted(self.__reachable_blocks)[-1]
            self.exit_blocks.append(last_block)
            
            if last_block == self.offset:
                for node in self.cfg.nodes():
                    if node != self.offset:
                        self.cfg.remove_node(node)
            

        # Some functions may share bytes and it can result in unreacheable
        # nodes, let's get rid of unreacheable nodes
        #
        for node, degree in self.cfg.degree(self.cfg.nodes()).iteritems():
            if node != self.offset and degree == 0:
                self.cfg.remove_node(node)
            
    def traverse(self, start_, depth = 0):
        if depth > 1000:
            return
        
        if start_ in self.__visited:
            return
        self.__visited.append(start_)

        def new_edge_and_traverse(from_, to_):
            self.__reachable_blocks.add(to_)
            self.add_edge(from_, to_)
            self.traverse(to_, depth + 1)

        addr = start_
        while True:
            if addr not in self.mem_map:
                if addr:
                    msg = "0x%x is not in mem_map. Bad disassembly?" % addr
                else:
                    msg = 'address is None. Bad disassembly?'                    
                raise FunctionAnalysisError(msg)
            
            op = self.mem_map[addr]
            
            self.blocks[start_].append(addr)
            if op['type'] == 'ret':
                self.exit_blocks.append(start_)
                return
                
            if op['type'] in ['cjmp', 'jmp', 'ujmp']:
                jmp_target = op.get('jump')
                if jmp_target:
                    new_edge_and_traverse(start_, jmp_target)
                if 'fail' in op:
                    new_edge_and_traverse(start_, op['fail'])
                return

            addr = op['next_addr']
            if addr in self.blocks and addr != self.offset:
                 new_edge_and_traverse(start_, addr)
                 return
        
    
    def find_longest_path(self):
        if len(self.cfg.nodes()) == 1:
            return [self.offset]
        
        longest_len = 0
        longest_path = None
        
        for addr in self.exit_blocks:
            for path in nx.all_simple_paths(self.cfg, self.offset, addr):
                if len(path) >= longest_len:
                    longest_len = len(path)
                    longest_path = path
                    
        return longest_path

    def get_path_random_walk(self):
        if len(self.cfg.nodes()) == 1:
            return [self.offset]
        
        longest_len = 0
        longest_path = None
        
        for i in xrange(30):
            path = self.random_walk()
            if len(path) >= longest_len:
                longest_len = len(path)
                longest_path = path
        return longest_path
            
    def get_paths(self):
        paths = []
        for addr in self.exit_blocks:
            paths.extend(nx.all_simple_paths(self.cfg, self.offset, addr))

        if not paths:
            paths = [[self.offset]]            
        return paths

    # TODO: ignore loops!
    def random_walk(self):
        path = [self.offset]
        block = self.offset

        while True:
            block_addr_list = self.blocks.get(block)

            if not block_addr_list:
                break
            
            op_addr = block_addr_list[-1]
            op = self.mem_map[op_addr]
            if op['type'] == 'jmp':
                if 'jump' in op:
                    target = op['jump']
                else:
                    break
            elif op['type'] == 'cjmp':
                target = random.choice([op.get('jump'), op.get('fail')])
            else:
                target = op['next_addr']

            path.append(target)
            # Pretty arbitrary number
            if len(path) > 300:
                break
            
            block = target
            
            if target in self.exit_blocks:
                break

        return path
    
