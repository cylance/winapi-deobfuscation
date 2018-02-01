import sys
import os
import argparse
import logging
import simplejson

from winapi_deobf import database
from winapi_deobf import radare
from winapi_deobf import util
from winapi_deobf import analysis
from winapi_deobf import esil

CACHE = 'cache'

def init_db(db_path):
    db = database.Database(db_path)
    db.create_tables()
    db.close()

    
def analyze(file_, db_path):
    db = database.Database(db_path)
    if db.file_exists(file_):
        logging.info("%s has already been analyzed" % file_)
        return
    
    db.add_file(file_)

    win_api_lookup = analysis.WinAPILookup()
    
    exe = radare.Radare(file_, CACHE)
    import_table = analysis.get_import_table(exe)
    functions = analysis.extract_functions(exe, import_table)

    # Traverse through functions in the executable
    for offset, function_ in functions.iteritems():
        # Skip trampoline jumps to IAT
        if function_.jmp_to_iat: continue

        # Building the graph
        cfg = analysis.CFG(function_)
        try:
            cfg.build_graph()
        except analysis.FunctionAnalysisError as e:
            logging.error(e)
            continue

        # Calculating the longest path
        n_edges = cfg.n_edges
        if n_edges < 100:
            path = cfg.find_longest_path()
        else:
            # If too many edges use random walk to
            # save time (but sacrifice precision)
            path = cfg.get_path_random_walk()


        # Instanciate ESIL symbolic execution engine
        vm = esil.ESILVM()
        # Execute the path
        # traverse through basic blocks
        for block_addr in path:
            # traverse through opcodes in each block
            for addr in cfg.blocks[block_addr]:
                op = cfg.mem_map[addr]
                # If it's not a call, execute normally
                if op['type'] not in ['ucall', 'call']:
                    if analysis.is_supported(op):
                        vm.execute(op['esil'])
                else:
                    # What if it IS a call?
                    # extract the call target
                    call_target = analysis.get_call_target(op, vm)
                    if call_target:
                        # if it's a call to an API, extract
                        # the relevant information
                        iat_entry = analysis.get_iat_entry(call_target, import_table, functions)

                        # If it's an API call
                        if iat_entry:
                            # Name of the API call, e.g. RtlFreeHeap
                            name = iat_entry['name']
                            # Number of arguments the API function takes in
                            n_args = win_api_lookup.look_up_number_of_arguments(name)
                            if n_args:
                                arguments = analysis.get_arguments_from_stack(n_args, vm, exe)
                                db.add_call(name, n_args, simplejson.dumps(arguments))
                    # we need to set some dummy return value to EAX
                    vm.update_eax()

    exe.quit()
    db.commit()
    db.close()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pe", help="Path to a PE executable")
    parser.add_argument("database", help="Path to SQLite3 database to store results")
    
    args = parser.parse_args()

    util.init_logging('DEBUG')

    if not os.path.exists(CACHE):
        os.mkdir(CACHE)

    init_db(args.database)
    analyze(args.pe, args.database)

if __name__ == '__main__':
    main()
