
def to_float(x):
    """Formating for index data."""
    if isinstance(x, str):
        x = x.replace(',','')
        if 'K' in x:
            p1, p2 = x.split('K')
            return float(p1) * 1000
        if 'M' in x:
            p1, p2 = x.split('M')
            return float(p1) * 1000000
        if 'B' in x:
            p1, p2 = x.split('B')
            return float(p1) * 1000000000
    return float(x) # already float!
