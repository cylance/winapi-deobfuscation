import argparse
import random
import simplejson

from winapi_deobf import database
from winapi_deobf import representation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlite', help="Path to the sqlite with the collected data")
    args = parser.parse_args()

    db = database.Database(args.sqlite)

    data_counter = {name:0 for name in representation.NAMES}
    n_max = 400

    X = []
    
    for call in db.get_calls():
        name = call['api_name']
        n_args = call['n_args']
        stack = simplejson.loads(call['stack'])
        
        if name.endswith('A') or name.endswith('W'):
            name = name[:-1]

        if name not in representation.NAMES: continue
        if data_counter[name] > n_max: continue

        data_counter[name] += 1

        x = []

        for i in range(len(stack)):
            repr = representation._repr(stack[i], i, name, n_args)
            x.append(repr)

        x.append(name)
        x.append(str(n_args))
        
        X.append(x)



    db.close()


    for i in xrange(30):
        random.shuffle(X)

    for x in X:
        print ','.join(x)


if __name__ == '__main__':
    main()
