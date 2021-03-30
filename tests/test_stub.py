import sys, os.path
ML_HOME = os.path.abspath(os.path.join("/opt/workspace", "ml_home"))
sys.path.insert(0, ML_HOME) # Add to path so can load our library
EXPERIMENT_HOME = os.path.abspath(os.path.join(ML_HOME, ".."))

from experiment import init_experiment
db_engine = init_experiment(EXPERIMENT_HOME)

# content of test_sample.py
def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5

