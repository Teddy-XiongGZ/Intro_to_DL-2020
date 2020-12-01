""" 
A number of utilities that help mask os differences
"""

import subprocess

def cmd(string):
    """ exec a cmd command and output to stdout """
    p = subprocess.Popen(string, shell=True)
    p.wait()
    return p.returncode