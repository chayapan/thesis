import pytest

def func(x):
    from data.snapshot import SET100, SET100_db_engine, make_index
    engine = SET100_db_engine()
    print(engine)
    return x + 1

def test_load_data_snapshot():
    assert func(4) == 5

def func_data_generators():
    from data.generator import plot_line, gd2df, add_noise, dgf10, dgf11, dgf1, dgf2, dgf3, dgf4, dgf5, dgf6, dgf7, dgf8, dgf9
    for g in [dgf10, dgf11, dgf1, dgf2, dgf3, dgf4, dgf5, dgf6, dgf7, dgf8, dgf9]:
        X,y = g()    
    return 1

def test_data_generators():
    assert func_data_generators() == 1