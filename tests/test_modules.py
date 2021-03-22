import pytest

def func(x):
    from data.snapshot import SET100, SET100_db_engine, make_index
    engine = SET100_db_engine()
    print(engine)
    return x + 1

def test_load_data_snapshot():
    assert func(4) == 5
