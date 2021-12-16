"""Historical financial data acquisition

>>> from funds import get_ticker_symbols
>>> tickers = get_ticker_symbols()
>>> len(tickers)
4039
>>> tickers[:3]
['XOG', 'FTA', 'GLADP']
"""

from funds.util import get_ticker_symbols
