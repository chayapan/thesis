from collections import namedtuple
from data.snapshot import sectors, industries

MarketYear = namedtuple('MarketYear', ['dt_start','dt_end'])
# period_start='2014-01-01'
# period_end='2014-12-31'
# df = df[period_start:period_end]
Yr2014 = MarketYear(dt_start='2014-01-01', dt_end='2014-12-31')
Yr2015 = MarketYear(dt_start='2015-01-01', dt_end='2015-12-31')
Yr2016 = MarketYear(dt_start='2016-01-01', dt_end='2016-12-31')
Yr2017 = MarketYear(dt_start='2017-01-01', dt_end='2017-12-31')
Yr2018 = MarketYear(dt_start='2018-01-01', dt_end='2018-12-31')
Yr2019 = MarketYear(dt_start='2019-01-01', dt_end='2019-12-31')
Yr2020 = MarketYear(dt_start='2020-01-01', dt_end='2020-12-31')

years = {   2014: Yr2014,
            2015: Yr2015,
            2016: Yr2016,
            2017: Yr2017,
            2018: Yr2018,
            2019: Yr2019,
            2020: Yr2020 }
