import os
from oldutils.datasets import ROOT_FOLDER


def set_root_folder():
    os.chdir(ROOT_FOLDER)


def init_context():
    import pandas as pd
    from warnings import simplefilter

    set_root_folder()
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
