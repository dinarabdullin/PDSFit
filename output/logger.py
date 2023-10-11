import sys
import traceback


class Logger(object):
    """A context manager that sends the output to the console and 
    simultaneously to an output file."""

    def __init__(self, filename='logfile.log'):
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