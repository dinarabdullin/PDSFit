import sys
import traceback


class Logger(object):
    '''
    A simple context manager that prints to the console and writes the same output to an file. 
    It also writes any exceptions to the file.
    Taken from:
    https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    To run:        
    sys.stdout = Logger()
    '''
    def __init__(self, filename="logfile.log"):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def isatty(self): 
        return False