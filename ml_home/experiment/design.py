
class Experiment:
    '''Base class for all experiment'''
    def __init__(self):
        self.summary = ""
    def __repr__(self):
        return """<%s summary="%s">""" % (self.__class__.__name__, self.summary)

class TimeSeriesClustering(Experiment):
    pass
