""" 
Wrapper commands
"""
import os
import subprocess
import sys

def cmd(string):
    """ exec a cmd command and output to stdout """
    if "colab" in vars() and colab:
      return get_ipython().system(string)
    else:
      p = subprocess.Popen(string, shell=True)
      p.wait()
      return p.returncode

def download(link):
    """ download a file using the link provided """
    if not os.path.exists("download"):
        cmd("mkdir download")
    cmd("cd download && wget -c {}".format(link))
    