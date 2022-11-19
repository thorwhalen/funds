
# funds
Historical finance data acquisition and caching


To install:	```pip install funds```

# Examples

## get a small set of tickers (offline, from a local file)

```python
from hedger import get_ticker_symbols
tickers = get_ticker_symbols()
len(tickers)
# 4039
'GOOG' in tickers
# True
```
