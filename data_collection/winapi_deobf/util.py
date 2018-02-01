import logging
import sqlite3
import hashlib
import zlib
import simplejson
import logging

def init_logging(level):
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def read_file(path):
    fd = open(path, 'rb')
    data = fd.read()
    fd.close()
    return data

def write_file(path, data):
    fd = open(path, 'wb')
    fd.write(data)
    fd.close()

def get_sha256(bytes_):
    return hashlib.sha256(bytes_).hexdigest()

def get_file_sha256(path):
    data = read_file(path)
    return get_sha256(data)

def compress(data):
    compressed = zlib.compress(data, 9)
    return sqlite3.Binary(compressed)

def pack_json(json):    
    return compress(simplejson.dumps(json)) if json else ''

def decompress(data):
    try:
        decomp = zlib.decompress(data)
    except zlib.error as e:
        return
    return decomp

def unpack_json(data):
    if not data:
        return {}
    data = decompress(data)
    return simplejson.loads(data) if data else {}

