# TODO:
# - SSE support
# - finish emulating all the ESIL commands
# - 64-bit support

import re
import logging
# logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
import inspect
import z3


INT_REGEX = re.compile(r'^-?\d+$')
HEX_REGEX = re.compile(r'^0x[A-Fa-f0-9]+$')
ESIL_FLAG_VALUE_REGEX = re.compile(r'$[0-9]+')

BITS = 32


def to_int(x):
    if INT_REGEX.match(x):
        return int(x)

    if HEX_REGEX.match(x):
        return int(x, 16)
    
    #raise RuntimeError('{0} - not an integer'.format(x))


def zero():
    return z3.BitVecVal(0, BITS)

def get32(x):
    # return x & 0xFFFFFFFF
    return x

def set32(x, y):
    # return (0xFFFFFFFF00000000 & x) | (y & 0xFFFFFFFF)
    return y

def get16(x):
    return x & 0xFFFF

def set16(x, y):
    return (0xFFFF0000 & x) | (y & 0xFFFF)

def get8h(x):
    return (x & 0xFF00) >> 8

def set8h(x, y):
    return (0xFFFF00FF & x) | ( y << 8 )

def get8l(x):
    return x & 0xFF

def set8l(x, y):
    return (0xFFFFFF00 & x) | ( y & 0xFF)

def ror(x, y):
    return z3.simplify(z3.RotateRight(x,y))

def rol(x,y):
    return z3.simplify(z3.RotateLeft(x,y))

def bool_to_int(x):
    return int(str(x) == 'k!1')

# https://stackoverflow.com/questions/1604464/twos-complement-in-python
def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is


def mem_key(addr, size):
    # addr_str = parse_addr_str(str(z3.simplify(addr)))
    addr_str = str(z3.simplify(addr))
    return '{0}:{1}'.format(size, addr_str)

def get_name_from_addr_str(addr):
    r_ebp = re.search(r'(\d+) \+ ebp', addr)
    if r_ebp:
        offset = twos_comp(int(r_ebp.group(1)), 32)
        if offset < 0:
            return 'var_%Xh' % -offset
        else:
            return 'arg_%Xh' % offset
        

REG_MAP  = {
    'rax': ('eax', get32, set32), # dummy
    'eax': ('eax', get32, set32),
    'ax': ('eax', get16, set16),
    'ah': ('eax', get8h, set8h),
    'al': ('eax', get8l, set8l),
    
    'ebx': ('ebx', get32, set32),
    'bx': ('ebx', get16, set16),
    'bh': ('ebx', get8h, set8h),
    'bl': ('ebx', get8l, set8l),
    
    'ecx': ('ecx', get32, set32),
    'cx': ('ecx', get16, set16),
    'ch': ('ecx', get8h, set8h),
    'cl': ('ecx', get8l, set8l),
    
    'edx': ('edx', get32, set32),
    'dx': ('edx', get16, set16),
    'dh': ('edx', get8h, set8h),
    'dl': ('edx', get8l, set8l),
    
    'esi': ('esi', get32, set32),
    'si': ('esi', get16, set16),
    'sil': ('esi', get8l, set8l),

    'edi': ('edi', get32, set32),
    'di': ('edi', get16, set16),
    'dil': ('edi', get8l, set8l),

    'esp': ('esp', get32, set32),
    'sp': ('esp', get16, set16),
    
    'ebp': ('ebp', get32, set32),
    'bp': ('ebp', get16, set16),
    
    'eip': ('eip', get32, set32)
}




class ESILVM(object):
    def __init__(self):
        self.registers = {
            'eax': z3.BitVec('eax', BITS),
            'ebx': z3.BitVec('ebx', BITS),
            'ecx': z3.BitVec('ecx', BITS),
            'edx': z3.BitVec('edx', BITS),
            'edi': z3.BitVec('edi', BITS),
            'esi': z3.BitVec('esi', BITS),
            'ebp': z3.BitVec('ebp', BITS),
            'esp': z3.BitVec('esp', BITS),
            'eip': z3.BitVec('eip', BITS)
        }

        """
        self.x86_flags = {
            'zf': zero(), 'pf': zero(),
            'sf': zero() , 'cf': zero(),
            'of': zero(), 'df': zero()
        }

        self.eflags = z3.BitVec('eflags', BITS)

        self.esil_flags = {
            '$z': zero(), '$b': zero(), '$c': zero(),
            '$o': zero(), '$p': zero(), '$r': zero(),
            '$s': zero()
        }

        self.segment_registers = {
            'cs': z3.BitVec('cs', BITS),
            'ds': z3.BitVec('ds', BITS),
            'ss': z3.BitVec('ss', BITS),
            'es': z3.BitVec('es', BITS),
            'fs': z3.BitVec('fs', BITS),
            'gs': z3.BitVec('gs', BITS)
        }
        """
        self.values = {}

        self.x86_stack = []
        self.__stack = []

        self.memory = {}

        self.instr_table = {
            "$": self.esil_interrupt,
            "==": self.esil_cmp,
            "<": self.esil_smaller,
            ">": self.esil_bigger,
            "<=": self.esil_smaller_equal,
            ">=": self.esil_bigger_equal,
            "?{": self.esil_if,
            "<<": self.esil_lsl,
            "<<=": self.esil_lsleq,
            ">>": self.esil_lsr,
            ">>=": self.esil_lsreq,
            ">>>>": self.esil_asr,
            ">>>>=": self.esil_asreq,
            ">>>": self.esil_ror,
            "<<<": self.esil_rol,
            "&": self.esil_and,
            "&=": self.esil_andeq,
            "}": self.esil_nop,
            "|": self.esil_or,
            "|=": self.esil_oreq,
            "!": self.esil_neg,
            "!=": self.esil_negeq,
            "=": self.esil_eq,
            "*": self.esil_mul,
            "*=": self.esil_muleq,
            "^": self.esil_xor,
            "^=": self.esil_xoreq,
            "+": self.esil_add,
            "+=": self.esil_addeq,
            "++": self.esil_inc,
            "++=": self.esil_inceq,
            "-": self.esil_sub,
            "-=": self.esil_subeq,
            "--": self.esil_dec,
            "--=": self.esil_deceq,
            "/": self.esil_div,
            "/=": self.esil_diveq,
            "%": self.esil_mod,
            "%=": self.esil_modeq,
            "=[]": self.esil_poke,
            "=[1]": self.esil_poke1,
            "=[2]": self.esil_poke2,
            "=[3]": self.esil_poke3,
            "=[4]": self.esil_poke4,
            "=[8]": self.esil_poke8,
            "|=[]": self.esil_mem_oreq,
            "|=[1]": self.esil_mem_oreq1,
            "|=[2]": self.esil_mem_oreq2,
            "|=[4]": self.esil_mem_oreq4,
            "|=[8]": self.esil_mem_oreq8,
            "^=[]": self.esil_mem_xoreq,
            "^=[1]": self.esil_mem_xoreq1,
            "^=[2]": self.esil_mem_xoreq2,
            "^=[4]": self.esil_mem_xoreq4,
            "^=[8]": self.esil_mem_xoreq8,
            "&=[]": self.esil_mem_andeq,
            "&=[1]": self.esil_mem_andeq1,
            "&=[2]": self.esil_mem_andeq2,
            "&=[4]": self.esil_mem_andeq4,
            "&=[8]": self.esil_mem_andeq8,
            "+=[]": self.esil_mem_addeq,
            "+=[1]": self.esil_mem_addeq1,
            "+=[2]": self.esil_mem_addeq2,
            "+=[4]": self.esil_mem_addeq4,
            "+=[8]": self.esil_mem_addeq8,
            "-=[]": self.esil_mem_subeq,
            "-=[1]": self.esil_mem_subeq1,
            "-=[2]": self.esil_mem_subeq2,
            "-=[4]": self.esil_mem_subeq4,
            "-=[8]": self.esil_mem_subeq8,
            "%=[]": self.esil_mem_modeq,
            "%=[1]": self.esil_mem_modeq1,
            "%=[2]": self.esil_mem_modeq2,
            "%=[4]": self.esil_mem_modeq4,
            "%=[8]": self.esil_mem_modeq8,
            "/=[]": self.esil_mem_diveq,
            "/=[1]": self.esil_mem_diveq1,
            "/=[2]": self.esil_mem_diveq2,
            "/=[4]": self.esil_mem_diveq4,
            "/=[8]": self.esil_mem_diveq8,
            "*=[]": self.esil_mem_muleq,
            "*=[1]": self.esil_mem_muleq1,
            "*=[2]": self.esil_mem_muleq2,
            "*=[4]": self.esil_mem_muleq4,
            "*=[8]": self.esil_mem_muleq8,
            "++=[]": self.esil_mem_inceq,
            "++=[1]": self.esil_mem_inceq1,
            "++=[2]": self.esil_mem_inceq2,
            "++=[4]": self.esil_mem_inceq4,
            "++=[8]": self.esil_mem_inceq8,
            "--=[]": self.esil_mem_deceq,
            "--=[1]": self.esil_mem_deceq1,
            "--=[2]": self.esil_mem_deceq2,
            "--=[4]": self.esil_mem_deceq4,
            "--=[8]": self.esil_mem_deceq8,
            "[]": self.esil_peek,
            "[*]": self.esil_peek_some,
            "=[*]": self.esil_poke_some,
            "[1]": self.esil_peek1,
            "[2]": self.esil_peek2,
            "[3]": self.esil_peek3,
            "[4]": self.esil_peek4,
            "[8]": self.esil_peek8,
            "STACK": self.r_anal_esil_dumpstack,
            "REPEAT": self.esil_repeat,
            "POP": self.esil_pop,
            "TODO": self.esil_todo,
            "GOTO": self.esil_goto,
            "BREAK": self.esil_break,
            "CLEAR": self.esil_clear,
            "DUP": self.esil_dup,
            "NUM": self.esil_num,
            "PICK": self.esil_pick,
            "RPICK": self.esil_rpick,
            "SWAP": self.esil_swap,
            "TRAP": self.esil_trap,
            "BITS": self.esil_bits,
        }
        
        self.__mem_var_cnt = 0
        self.__unk_var_cnt = 0

    def update_eax(self):
        self.set_reg('eax', z3.BitVec('ret', 32))
        
    def pop(self):
        return self.__stack.pop()

    def push(self, x):
        self.__stack.append(x)

    def set_reg(self, name, value):
        key, _, set_ = REG_MAP[name]
        x = self.registers[key]
        self.registers[key] = set_(x, value)

    def get_reg(self, name):
        key, get_, _ = REG_MAP[name]
        return  get_(self.registers[key])
    
    def write_mem(self, addr, value, size):
        if isinstance(addr, str):
            key = addr
        else:
            key = mem_key(addr, size)
        self.memory[key] = value

    def arg_or_var(self, addr):
        name = get_name_from_addr_str(addr)
        if name:
            return z3.BitVec(name, 32)
        return self.__new_mem_var()
        
    def read_mem(self, addr, size):
        key = mem_key(addr, size)
        if key not in self.memory:
            self.memory[key] = self.arg_or_var(key)
        return self.memory[key]


    def __is_register(self, x):
        return isinstance(x, str) and x in REG_MAP

    """
    def __is_x86_flag(self, x):
        return isinstance(x, str) and x in self.x86_flags

    def __is_ESIL_flag(self, x):
        return isinstance(x, str) and x.startswith('$') and x != '$' and x != '$$'

    def __is_segment_register(self, x):
        return isinstance(x, str) and x in self.segment_registers
    """

    def __is_eflags(self, x):
        return isinstance(x, str) and x == 'eflags'

    """
    def __is_ESIL_flag_value(self, x):
        return isinstance(x, str) and ESIL_FLAG_VALUE_REGEX.match(x)
    """
    
    def __resolve(self, x):
        if self.__is_register(x):
            return self.get_reg(x)

        if isinstance(x, str):
            return zero()

        return x
        """
        if self.__is_ESIL_flag(x):
            return self.esil_flags.get(x, zero())
        if self.__is_x86_flag(x):
            return self.x86_flags[x]
        if self.__is_eflags(x):
            return self.eflags
        if self.__is_segment_register(x):
            return self.segment_registers.get(x, zero())
        """
        # return x

    def __new_unk_var(self):
        x = z3.BitVec('unk{0}'.format(self.__unk_var_cnt), 32)
        self.__unk_var_cnt += 1
        return x

    def __new_mem_var(self):
        x = z3.BitVec('mem{0}'.format(self.__mem_var_cnt), 32)
        self.__mem_var_cnt += 1
        return x

    def __get_2_operands(self):
        x = self.__resolve(self.pop())
        y = self.__resolve(self.pop())
        return x, y

    def __binop(self, cmd):
        x, y = self.__get_2_operands()
        self.push(cmd(x, y))

    def __binop_eq(self, cmd):
        dst = self.pop()
        x = self.__resolve(self.pop())
        self.__set(dst, cmd(self.__resolve(dst), x))

    def __set(self, dst, x):
        if self.__is_register(dst):
            self.set_reg(dst, x)
                            
    def __unop(self, cmd):
        x = self.__resolve(self.pop())
        self.push(cmd(x))

    def __unop_eq(self, cmd):
        dst = self.pop()
        if self.__is_register(dst):
            self.set_reg(dst, cmd(self.get_reg(dst)))

    def __mem_binop_eq(self, cmd, size):
        addr, x, y = self.__get_mem_params()
        self.write_mem(addr, cmd(x, y), size)

    def __get_mem_params(self):
        addr = self.__resolve(self.pop())
        x = self.__resolve(self.pop())
        y = self.read_mem(addr, 4)
        return addr, x, y

    def __mem_unop_eq(self, cmd, size):
        addr = self.__resolve(self.pop())
        x = self.read_mem(addr, size)
        self.write_mem(addr, cmd(x), size)
    
    def __cmp(self, x, y):
        return bool_to_int((x == y))

    def __smaller(self, x, y):
        return bool_to_int((x < y))

    def __bigger(self, x, y):
        return bool_to_int((x > y))

    def __smaller_equal(self, x, y):
        return bool_to_int((x <= y))

    def __bigger_equal(self, x, y):
        return bool_to_int((x >= y))

    def __lsl(self, x, y):
        return x << y

    def __lsr(self, x, y):
        return x >> y

    def __and(self, x, y):
        return x & y

    def __or(self, x, y):
        return x | y

    def __neg(self, x):
        return -x

    def __mul(self, x, y):
        return x * y

    def __xor(self, x, y):
        return x ^ y

    def __add(self, x, y):
        return x + y

    def __inc(self, x):
        return x + 1

    def __sub(self, x, y):
        return x - y

    def __dec(self, x):
        return x - 1

    def __div(self, x, y):
        return x / y

    def __mod(self, x, y):
        return x % y

        
    def execute(self, expr):
        expr = expr.encode('utf-8')


        for op in expr.split(','):
            val = to_int(op)
            if val:
                self.push(z3.BitVecVal(val, 32))
            elif op in self.instr_table:
                self.instr_table[op]()
                
            elif op == '':
                # Sometimes a value might be absent, set to 0 as a
                # temporary workaround, probably a bug in ESIL
                self.push(zero())
            else:
                # Otherwise it's either register or
                # a flag
                self.push(op)
                

    # Commands
    def esil_interrupt(self):
        # TODO:
        pass

    def esil_cmp(self):
        self.__binop(self.__cmp)

    def esil_smaller(self):
        self.__binop(self.__smaller)
        
    def esil_bigger(self):
        self.__binop(self.__bigger)

    def esil_smaller_equal(self):
        self.__binop(self.__smaller_equal)
        
    def esil_bigger_equal(self):
        self.__binop(self.__bigger_equal)

    def esil_if(self):
        # raise RuntimeError("?{ (esil_if): not implemented!")
        # TODO: finish
        self.pop()
        
    def esil_lsl(self):
        self.__binop(self.__lsl)

    def esil_lsleq(self):
        self.__binop_eq(self.__lsl)

    def esil_lsr(self):
        self.__binop(self.__lsr)
        
    def esil_lsreq(self):
        self.__binop_eq(self.__lsr)
        
    def esil_asr(self):
        raise RuntimeError(">>>> (esil_asr): not implemented!")

    def esil_asreq(self):
        raise RuntimeError(">>>>= (esil_asreq): not implemented!")
        
    def esil_ror(self):
        self.__binop(ror)

    def esil_rol(self):
        self.__binop(rol)
        
    def esil_and(self):
        self.__binop(self.__and)
        
    def esil_andeq(self):
        self.__binop_eq(self.__and)

    def esil_nop(self):
        pass

    def esil_or(self):
        self.__binop(self.__or)
        
    def esil_oreq(self):
        self.__binop_eq(self.__or)

    def esil_neg(self):
        self.__unop(self.__neg)
        
    def esil_negeq(self):
        self.__unop_eq(self.__neg)
        
    def esil_eq(self):
        dst = self.pop()
        src = self.pop()

        # We want to keep ebp symbolic value
        if self.__is_register(dst) and dst == 'ebp' and \
           self.__is_register(src) and src == 'esp':
            return

        if self.__is_eflags(dst):
            return
        
        x = self.__resolve(src)
        self.__set(dst, x)
            
    def esil_mul(self):
        self.__binop(self.__mul)
        
    def esil_muleq(self):
        self.__binop_eq(self.__mul)

    def esil_xor(self):
        self.__binop(self.__xor)
        
    def esil_xoreq(self):
        self.__binop_eq(self.__xor)
        
    def esil_add(self):
        self.__binop(self.__add)
        
    def esil_addeq(self):
        self.__binop_eq(self.__add)
        
    def esil_inc(self):
        self.__unop(self.__inc)
        
    def esil_inceq(self):
        self.__unop_eq(self.__inc)
        
    def esil_sub(self):
        self.__binop(self.__sub)
        
    def esil_subeq(self):
        self.__binop_eq(self.__sub)

    def esil_dec(self):
        self.__unop(self.__dec)
        
    def esil_deceq(self):
        self.__unop_eq(self.__dec)
        
    def esil_div(self):
        self.__binop(self.__div)

    def esil_diveq(self):
        self.__binop_eq(self.__div)

    def esil_mod(self):
        self.__binop(self.__mod)

    def esil_modeq(self):
        self.__binop_eq(self.__mod)
        
    def esil_poke(self, size=4):
        #raise RuntimeError("=[] (esil_poke): not implemented!")
        dst = self.pop()
        x = self.__resolve(self.pop())
        if self.__is_register(dst):
            if dst == 'esp':
                self.x86_stack.append(x)
            else:
                addr = self.get_reg(dst)
                self.write_mem(addr, x, size)
        else:
            self.write_mem(dst, x, size)

    def esil_poke1(self):
        # raise RuntimeError("=[1] (esil_poke1): not implemented!")
        self.esil_poke(1)

    def esil_poke2(self):
        # raise RuntimeError("=[2] (esil_poke2): not implemented!")
        self.esil_poke(2)

    def esil_poke3(self):
        raise RuntimeError("=[3] (esil_poke3): not implemented!")

    def esil_poke4(self):
        # raise RuntimeError("=[4] (esil_poke4): not implemented!")
        self.esil_poke(4)
        
    def esil_poke8(self):
        raise RuntimeError("=[8] (esil_poke8): not implemented!")

    def esil_poke_some(self):
        raise RuntimeError("=[*] (esil_poke_some): not implemented!")
    
    def esil_mem_oreq(self, size=4):
        self.__mem_binop_eq(self.__or, size)

    def esil_mem_oreq1(self):
        self.esil_mem_oreq(1)

    def esil_mem_oreq2(self):
        self.esil_mem_oreq(2)

    def esil_mem_oreq4(self):
        self.esil_mem_oreq(4)

    def esil_mem_oreq8(self):
        raise RuntimeError("|=[8] (esil_mem_oreq8): not implemented!")

    def esil_mem_xoreq(self, size=4):
        self.__mem_binop_eq(self.__xor, size)

    def esil_mem_xoreq1(self):
        self.esil_mem_xoreq(1)

    def esil_mem_xoreq2(self):
        self.esil_mem_xoreq(2)

    def esil_mem_xoreq4(self):
        self.esil_mem_xoreq(4)

    def esil_mem_xoreq8(self):
        raise RuntimeError("^=[8] (esil_mem_xoreq8): not implemented!")

    def esil_mem_andeq(self, size=4):
        self.__mem_binop_eq(self.__and, size)
        
    def esil_mem_andeq1(self):
        self.esil_mem_andeq(1)

    def esil_mem_andeq2(self):
        self.esil_mem_andeq(2)

    def esil_mem_andeq4(self):
        self.esil_mem_andeq(4)

    def esil_mem_andeq8(self):
        raise RuntimeError("&=[8] (esil_mem_andeq8): not implemented!")

    def esil_mem_addeq(self, size=4):
        self.__mem_binop_eq(self.__add, size)
        
    def esil_mem_addeq1(self):
        self.esil_mem_addeq(1)

    def esil_mem_addeq2(self):
        self.esil_mem_addeq(2)

    def esil_mem_addeq4(self):
        self.esil_mem_addeq(4)

    def esil_mem_addeq8(self):
        raise RuntimeError("+=[8] (esil_mem_addeq8): not implemented!")

    def esil_mem_subeq(self, size=4):
        self.__mem_binop_eq(self.__sub, size)
        
    def esil_mem_subeq1(self):
        self.esil_mem_subeq(1)

    def esil_mem_subeq2(self):
        self.esil_mem_subeq(2)

    def esil_mem_subeq4(self):
        self.esil_mem_subeq(4)

    def esil_mem_subeq8(self):
        raise RuntimeError("-=[8] (esil_mem_subeq8): not implemented!")

    def esil_mem_modeq(self, size=4):
        self.__mem_binop_eq(self.__mod, size)
        
    def esil_mem_modeq1(self):
        self.esil_mem_modeq(1)
        
    def esil_mem_modeq2(self):
        self.esil_mem_modeq(2)

    def esil_mem_modeq4(self):
        self.esil_mem_modeq(4)

    def esil_mem_modeq8(self):
        raise RuntimeError("%=[8] (esil_mem_modeq8): not implemented!")

    def esil_mem_diveq(self, size=4):
        self.__mem_binop_eq(self.__div, size)

    def esil_mem_diveq1(self):
        self.esil_mem_diveq(1)

    def esil_mem_diveq2(self):
        self.esil_mem_diveq(2)

    def esil_mem_diveq4(self):
        self.esil_mem_diveq(4)

    def esil_mem_diveq8(self):
        raise RuntimeError("/=[8] (esil_mem_diveq8): not implemented!")

    def esil_mem_muleq(self, size=4):
        self.__mem_binop_eq(self.__mul, size)

    def esil_mem_muleq1(self):
        self.esil_mem_muleq(1)

    def esil_mem_muleq2(self):
        self.esil_mem_muleq(2)

    def esil_mem_muleq4(self):
        self.esil_mem_muleq(4)

    def esil_mem_muleq8(self):
        raise RuntimeError("*=[8] (esil_mem_muleq8): not implemented!")

    def esil_mem_inceq(self, size=4):
        self.__mem_unop_eq(self.__inc, size)
        
    def esil_mem_inceq1(self):
        self.esil_mem_inceq(1)

    def esil_mem_inceq2(self):
        self.esil_mem_inceq(2)

    def esil_mem_inceq4(self):
        self.esil_mem_inceq(4)

    def esil_mem_inceq8(self):
        raise RuntimeError("++=[8] (esil_mem_inceq8): not implemented!")

    def esil_mem_deceq(self, size=4):
        self.__mem_unop_eq(self.__dec, size)

    def esil_mem_deceq1(self):
        self.esil_mem_deceq(1)

    def esil_mem_deceq2(self):
        self.esil_mem_deceq(2)

    def esil_mem_deceq4(self):
        self.esil_mem_deceq(4)

    def esil_mem_deceq8(self):
        raise RuntimeError("--=[8] (esil_mem_deceq8): not implemented!")

    def esil_peek(self, size=4):
        # raise RuntimeError("[] (esil_peek): not implemented!")
        addr = self.pop()

        if self.__is_register(addr):
            if addr  == 'esp':
                if self.x86_stack:
                    x = self.x86_stack.pop()
                else:
                    x = self.__new_unk_var()
                self.push(x)
            else:
                x = self.read_mem(self.get_reg(addr), size)
                self.push(x)
        else:
            x = self.read_mem(addr, size)
            self.push(x)
            

    def esil_peek_some(self):
        raise RuntimeError("[*] (esil_peek_some): not implemented!")

    def esil_pokesome(self):
        raise RuntimeError("=[*] (esil_poke_some): not implemented!")

    def esil_peek1(self):
        # raise RuntimeError("[1] (esil_peek1): not implemented!")
        self.esil_peek(1)

    def esil_peek2(self):
        # raise RuntimeError("[2] (esil_peek2): not implemented!")
        self.esil_peek(2)

    def esil_peek3(self):
        raise RuntimeError("[3] (esil_peek3): not implemented!")

    def esil_peek4(self):
        # raise RuntimeError("[4] (esil_peek4): not implemented!")
        self.esil_peek(4)
        
    def esil_peek8(self):
        raise RuntimeError("[8] (esil_peek8): not implemented!")
        
    def r_anal_esil_dumpstack(self):
        raise RuntimeError("STACK (r_anal_esil_dumpstack): not implemented!")

    def esil_repeat(self):
        # raise RuntimeError("REPEAT (esil_repeat): not implemented!")
        pass

    def esil_pop(self):
        raise RuntimeError("POP (esil_pop): not implemented!")

    def esil_todo(self):
        raise RuntimeError("TODO (esil_todo): not implemented!")

    def esil_goto(self):
        raise RuntimeError("GOTO (esil_goto): not implemented!")

    def esil_break(self):
        raise RuntimeError("BREAK (esil_break): not implemented!")

    def esil_clear(self):
        raise RuntimeError("CLEAR (esil_clear): not implemented!")

    def esil_dup(self):
        raise RuntimeError("DUP (esil_dup): not implemented!")

    def esil_num(self):
        raise RuntimeError("NUM (esil_num): not implemented!")

    def esil_pick(self):
        raise RuntimeError("PICK (esil_pick): not implemented!")

    def esil_rpick(self):
        raise RuntimeError("RPICK (esil_rpick): not implemented!")

    def esil_swap(self):
        raise RuntimeError("SWAP (esil_swap): not implemented!")

    def esil_trap(self):
        raise RuntimeError("TRAP (esil_trap): not implemented!")

    def esil_bits(self):
        raise RuntimeError("BITS (esil_bits): not implemented!")
    

