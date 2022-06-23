from preprocess import preprocess_tess_data
import lightkurve
import pandas as pd

df=pd.read_csv("tois.csv",skiprows=4)

for tic in df["TIC"][:10]:
  try:
    preprocess_tess_data(tic)
  except lightkurve.utils.LightkurveError as e:
    print(e.str())

  # Iterate over a tess id from 10-100, using the files from running preprocess.py
  # Add checks for if the id fails, continue to find one that works
  # use pandas
  # df=pd.read_csv("tois.csv",skiprows=4)
  # Maybe for later clean up files on failure
# 1: iterate over a number of TOI (>10 and <100) (OR the TIC)

# 2: call preprocess_tess_data()
# 3: use TOI as an argument

# 4: check inputs 

# 5: return an error message? Or warning that is descriptive

# 6: use some kind of logic to let it keep working


# What to work with sorta
# $ py
# Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# >>> import numpy as np
# >>> np.load("code\2016376984_local_flux.npy")
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "", line 407, in load
#     fid = stack.enter_context(open(os_fspath(file), "rb"))
# FileNotFoundError: [Errno 2] No such file or directory: 'code\x816376984_local_flux.npy'
# >>>
# use this file or one of the global ones
# >>> np.load("2016376984_global_flux.npy")
# array([ 1.14985872,  1.10539097,  1.07428765,  1.2089824 ,  0.90175793,
#         0.94532459,  1.23639776,  0.76785173,  0.98411165,  0.79261856,
#         1.03632109,  1.01241508,  0.98589726,  1.1248106 ,  1.10727019,
#         1.14571055,  1.08385317,  1.18391857,  1.28848409,  0.83681323,
# ])
# >>> import matplotlib.pyplot as plt
# >>> plt.show()
# >>> plt.("2016376984_global_flux")
#   File "<stdin>", line 1
#     plt.("2016376984_global_flux")
#         ^
# SyntaxError: invalid syntax
# >>>