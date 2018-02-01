import sqlite3

BATCH_SIZE = 1000

def dict_factory(cursor, row):
    d = {}
    for idx,col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

class Database:
    def __init__(self, filename):
        self.__conn = sqlite3.connect(filename)
        self.__conn.row_factory = dict_factory
        self.__cursor = self.__conn.cursor()
        
    def commit(self):
        self.__conn.commit()

    def close(self):
        self.__cursor.close()
        self.__conn.close()

    def create_tables(self):
        self.__cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
        path TEXT UNIQUE
        )""")

        self.__cursor.execute("""
        CREATE TABLE IF NOT EXISTS calls (
        api_name INT,
        n_args,
        stack TEXT
        )""")
                
        self.commit()

    def add_file(self, path):
        self.__cursor.execute("""
        INSERT INTO files VALUES (?)
        """, (path,))

    def file_exists(self, path):
        self.__cursor.execute("""
        SELECT * FROM files WHERE path = ?
        """, (path,))
        return self.__cursor.fetchone() != None

    def add_call(self, api_name, n_args, stack):
        self.__cursor.execute("""
        INSERT INTO calls VALUES (?, ?, ?)
        """, (api_name, n_args, stack))

    def query(self, sql_query):
        self.__cursor.execute(sql_query)
        return self.__cursor.fetchall()

    def get_calls(self):
        self.__cursor.execute("SELECT * FROM calls")
        return self.__cursor
