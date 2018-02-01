# Data Collection

## Dependencies:
* Z3 with Python bindings enabled https://github.com/Z3Prover/z3
* NetworkX https://networkx.github.io/
* radare2 https://github.com/radare/radare2 + r2pipe (can be installed via pip or https://github.com/radare/radare2-r2pipe)

## Usage:

### process-exe.py

```
usage: process-exe.py [-h] pe database

positional arguments:
  pe          Path to a PE executable
  database    Path to SQLite3 database to store results

```

Takes in a PE file, extracts API calls and puts them into a SQLite3 database. This is what Algorithm 1 describes in the paper.

### gen-training-data.py

```
usage: gen-training-data.py [-h] sqlite

positional arguments:
  sqlite      Path to the sqlite with the collected data

```

Reads the SQLite3 database with API call data, applies the data representation rules described in Section IV.D of the paper. And outputs a list of the following format:

```
stack_value_1, stack_value_2, ..., stack_value_n, name_of_the_api_function, number_of_arguments
``

### training-dataset.txt

If you don't feel like generating the data yourself, take our dataset of 9837 API call entries and proceed straight to the experiments.