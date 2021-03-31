import sys, os.path
# setup environment
ML_HOME = os.path.abspath(os.path.join("/opt/workspace", "ml_home"))
sys.path.insert(0, ML_HOME) # Add to path so can load our library
EXPERIMENT_HOME = os.path.abspath(os.path.join(ML_HOME, ".."))

from experiment import init_experiment
db_engine = init_experiment(EXPERIMENT_HOME)

# content of test_stub.py
def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5



@pytest.fixture
def error_fixture():
    assert 0


def test_ok():
    print("ok")


def test_fail():
    assert 0


def test_error(error_fixture):
    pass


def test_skip():
    pytest.skip("skipping this test")


def test_xfail():
    pytest.xfail("xfailing this test")


@pytest.mark.xfail(reason="always xfail")
def test_xpass():
    pass
