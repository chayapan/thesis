import sys, os, os.path
from sqlalchemy import create_engine

# database
def db_engine():
    engine = create_engine('postgresql://datauser:1234@172.18.0.1:5432/stockdb', echo=False)
    return engine
    
def init_experiment(EXPERIMENT_HOME):
    DATA_HOME = os.path.abspath(os.path.join(EXPERIMENT_HOME,"dataset")) # Dataset location
    os.environ["EXPERIMENT_HOME"] = EXPERIMENT_HOME
    os.environ["DATA_HOME"] = DATA_HOME
    sys.path.insert(0, EXPERIMENT_HOME)
    os.chdir(EXPERIMENT_HOME) # Change working directory to experiment workspace
    db_conn = db_engine() # Get database engine
    print("Experiment Home: ", os.path.abspath(os.curdir), "; Data Home:", DATA_HOME, "; \nDatabase:", str(db_conn))
    return db_conn