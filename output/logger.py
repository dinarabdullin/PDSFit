import traceback
import sys

class Logger(object):
    '''
    Write all results being printed on stdout from the python source to file to the logfile.
    Taken from:
    https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    '''
    def __init__(self, log_name="logfile.log"):
        self.terminal = sys.stdout
        self.log_name = log_name

    def write(self, message):
        with open(self.log_name, "a", encoding = 'utf-8') as self.log:            
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        pass

class ContextManager(object):
    '''
    A simple context manager that prints to the console and writes the same output to an file. 
    It also writes any exceptions to the file.
    Taken from:
    https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    To run:        
    sys.stdout = ContextManager()
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