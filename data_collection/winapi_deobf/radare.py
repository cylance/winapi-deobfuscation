import os
import r2pipe
import util
import logging

class Radare:
    def __init__(self, path, cache_path):
        self.exe_path = path
        self.__r2p = r2pipe.open(path)
        self.__cache_file = os.path.join(cache_path, os.path.basename(path)+'.cache')
        self.__data = {}
        self.__need_save_cache = False
        self.__cached = self.__check_cache()
        self.__analyzed = False
        if not self.__cached:
            self.analyze()

    def __save_cache(self):
        if self.__need_save_cache:
            logging.debug("Saving cache...")
            packed = util.pack_json(self.__data)
            util.write_file(self.__cache_file, packed)

    def __check_cache(self):
        if os.path.exists(self.__cache_file):
            self.__data = util.unpack_json(util.read_file(self.__cache_file))
            return True
        return False

    def quit(self):
        self.__save_cache()
        self.__r2p.quit()
        
    def cmdj(self, command, suffix = ''):
        key = '{0}{1}'.format(command, suffix)

        if key in self.__data:
            logging.debug("{0} - found cached".format(command))
            return self.__data.get(key)
        else:
            logging.debug("{0} - not cached, executing...".format(command))
            if not self.__analyzed:
                self.analyze()
                
            try:
                res = self.__r2p.cmdj(command)
            except AttributeError as e:
                logging.error(str(e))
                return
            self.__data[key] = res
            self.__need_save_cache = True
            return res
                
    def cmd(self, command):
        try:
            logging.debug('> {0}'.format(command))
            self.__r2p.cmd(command)
        except AttributeError as e:
            logging.error("Couldn't execute {0}".format(command))
            return False
        return True
    
    def analyze(self):
        # if not self.__cached:
        # self.__need_save_cache = True
        # logging.debug("Cache not found, executing 'aac'")
        self.cmd('aac;')
        self.__analyzed = True
        # else:
        # logging.debug("Cached")

    def get_import_table(self):
        return self.cmdj('iij')
    
    def get_functions(self):
        return self.cmdj('aflj')

    def get_sections(self):
        return self.cmdj('iSj')
        
    def get_disassembly(self, offset, size=None):
        if self.cmd('s 0x%x;' % offset):
            # return self.cmdj('pdfj', hex(offset))
            # return self.cmdj('pDj {0}'.format(size), hex(offset))
            return self.cmdj('pdrj', hex(offset))
            
    def get_file_info(self):
        return self.cmdj('ij')

    def get_entry_point(self):
        return self.cmdj('iej')[0]
