"""Get company info joined with historical metrics and save to jsons"""

# TODO: Can do better by using merge/join in various places (instead of loops)

from dol import KvReader
from functools import cached_property, lru_cache
from io import BytesIO
import pandas as pd
from py2store import FilesOfZip, wrap_kvs, filt_iter, add_ipython_key_completions
from funds.util import proj_files


def to_camel_case(s: str):
    """
    >>> to_camel_case('This, (my friend) is camel_case!')
    'thisMyFriendIsCamelCase'
    """
    t = ''.join(x for x in s.title() if x.isalnum())
    return t[0].lower() + t[1:]


SIMFIN_PLUS_ZIP_PATH = str(proj_files.parent / 'misc/local/simfin+_2021-07-09.zip')


@add_ipython_key_completions
@wrap_kvs(
    key_of_id=lambda k: k[len('data_2021-07-09/') :],
    id_of_key=lambda k: 'data_2021-07-09/' + k,
)
@wrap_kvs(obj_of_data=lambda x: pd.read_csv(BytesIO(x), delimiter=';'))
@filt_iter(filt=lambda x: x.endswith('.csv'))
class SimfinZippedCsvs(FilesOfZip):
    """Dataframes from a csv files within a zip file"""


# z = ZippedCsvs('local/simfin+_2021-07-09.zip')


@lru_cache(maxsize=1)
def get_simfin_src_store(simfin_zip_path=SIMFIN_PLUS_ZIP_PATH):
    return SimfinZippedCsvs(simfin_zip_path)


@lru_cache(maxsize=1)
def get_dflt_info():
    from funds.scrap.company_info_prep import get_companies_info

    return get_companies_info()


@lru_cache(maxsize=1)
def get_nums(simfin_zip_path=SIMFIN_PLUS_ZIP_PATH):
    z = get_simfin_src_store(simfin_zip_path)
    share_prices = z['us-shareprices-daily.csv']
    derived_share_prices = z['us-derived-shareprices-daily.csv']

    nums = pd.merge(
        share_prices, derived_share_prices, on=['Ticker', 'SimFinId', 'Date']
    )

    return nums


# Note: Used to be called get_default_history_df
@lru_cache(maxsize=1)
def get_nums_indices_for_ticker(simfin_zip_path=SIMFIN_PLUS_ZIP_PATH):
    nums = get_nums(simfin_zip_path)
    return nums.groupby('Ticker').indices


def _get_from_callback_if_not_df(df):
    if isinstance(df, pd.DataFrame):
        return df
    elif callable(df):
        return df()
    else:
        raise TypeError(f'Unknown type for df: {df}')


class TickerHistory(KvReader):
    def __init__(self, history_df=get_nums_indices_for_ticker):
        history_df = _get_from_callback_if_not_df(history_df)
        df = history_df.copy()
        h = TickerInfoJson()
        df.columns = list(map(str.lower, h.history.history_df.columns))
        self.history_df = df

    #         self.nums_idx_for_ticker = nums.groupby('Ticker').indices

    @cached_property
    def nums_idx_for_ticker(self):
        return self.history_df.groupby('ticker').indices

    def __iter__(self):
        yield from self.nums_idx_for_ticker

    def __getitem__(self, ticker):
        return self.history_df.loc[self.nums_idx_for_ticker[ticker]]


class TickerInfoJson(KvReader):
    def __init__(
        self,
        history_df=get_nums_indices_for_ticker,
        info_df=None,
        include=None,
        key_fields=('ticker', 'simfinid'),
        exclude=None,
    ):
        history_df = _get_from_callback_if_not_df(history_df)
        self.history = TickerHistory(history_df)
        if info_df is None:
            info_df = get_dflt_info()
        if include is None:
            include = self.history.history_df.columns
        if exclude is None:
            exclude = key_fields
        self.include = [x for x in include if x not in exclude]  # sorted difference
        self.key_fields = key_fields
        self.exclude = exclude  # for the record
        self.info_df = info_df

    @cached_property
    def tickers(self):
        return sorted(set(self.history) | set(self.info_df.index))

    def __iter__(self):
        yield from self.tickers

    def df_for(self, k):
        return self.history.get(k, None)

    def __getitem__(self, k):
        df = self.df_for(k)
        if df is None:
            return None
        d = {}
        # keys
        for key_field in self.key_fields:
            t = df[key_field].unique()
            assert len(t) <= 1, (
                f"key fields values should be unique -- it wasn't the case "
                f'for the {key_field} (for {k})'
            )
            d.update({key_field: t[0]})
        # info
        info = self.info_df.T.get('AAPL', None)
        if info is None:
            info = {}
        else:
            info = info.to_dict()
        d.update(info)
        # metrics
        for metric_field in self.include:
            d.update({metric_field: df[metric_field].tolist()})
        return d


import json
import numpy


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


# t - dict(h)
# with open('company_ticker_info_and_metrics.json', 'w') as fp:
#     json.dump(t, fp, cls=MyEncoder)

# rootdir = "/Users/Thor.Whalen/Dropbox/_odata/finance/company_info_and_metric_jsons"
from dol import Files, wrap_kvs, Pipe
from functools import partial

numpy_aware_json_dump = partial(json.dumps, cls=MyEncoder)
to_numpy_aware_json = Pipe(numpy_aware_json_dump, str.encode)
JsonFiles = wrap_kvs(
    Files,
    data_of_obj=to_numpy_aware_json,
    obj_of_data=json.loads,
    key_of_id=lambda x: x[: -len('.json')],
    id_of_key=lambda x: x + '.json',
)

# h = TickerInfoJson()
# self = h
# len(h.tickers)
