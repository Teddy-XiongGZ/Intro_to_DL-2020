""" 
Wrapper commands
"""
import re
import os
import sys
import subprocess

def cmd(string):
    """ exec a command and output to stdout """
    if "colab" in vars() and colab:
      return get_ipython().system(string)
    else:
      p = subprocess.Popen(string, shell=True)
      p.wait()
      return p.returncode

def download(link):
    """ download a file to ./download/ using the provided link """
    if not os.path.exists("download"):
        os.mkdir("download")
    if not os.path.exists(os.path.join("./download", re.search("/([\w.]+)$", link).group(1))):
      cmd("cd download && wget -c {}".format(link))
    
