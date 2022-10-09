import os
import fxcmpy

from constants import *


"""
https://fxcm-api.readthedocs.io/en/latest/fxcmpy.html
"""


fxcm = fxcmpy.fxcmpy(access_token=FXCM_TOKEN, log_level='error', log_file=os.getcwd() +'\log.txt')

if __name__ == "__main__":

    pass
