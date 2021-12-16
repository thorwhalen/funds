"""Historical financial data acquisition

>>> from funds import get_ticker_symbols
>>> tickers = get_ticker_symbols()
>>> len(tickers)
4039
>>> 'GOOG' in tickers
True
"""

from funds.util import get_ticker_symbols
