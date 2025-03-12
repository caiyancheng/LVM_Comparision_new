import os.path
from PIL import Image
import numpy as np


class js_report:

    def __init__(self, file_name):

        self.name_ind = 0  # to generate unique file names        
        self.path, file = os.path.split(file_name)

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        self.js_file_name = file_name
        self.fh = open(file_name, "w")

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def print(self, str):
        self.fh.write( str )

    def println(self, str):
        self.fh.write( str + "\n\n" )     

    def close( self ):

        self.fh.close()
        print( "JS file generated in file://{}".format(os.path.abspath(self.js_file_name)) )

