from itertools import cycle
from functools import reduce, lru_cache
from operator import and_, attrgetter
from collections import Counter

import pandas as pd
import numpy as np

from i2 import Pipe

from funds.scrap.company_info_w_historical_metrics import (
    get_simfin_src_store,
    JsonFiles,
)
from funds.scrap.company_info_prep import get_companies_info


def make_company_info_and_metrics_jsons_and_save_them(save_root_dir):
    def gen():
        s = JsonFiles(save_root_dir)
        for group, sources in data_groups.items():
            df = merge_group(sources)
            for ticker_data in group_gather_and_complete_with_info(df):
                if ticker := ticker_data.get('ticker'):
                    s[ticker] = dict(ticker_data, data_group=group)
                else:
                    yield ticker_data  # to be collected for error checking
    return list(gen())


def get_src_store(src_store=None):
    if src_store is None:
        return get_simfin_src_store()
    else:
        return src_store


def df_info(name, z=None):
    z = get_src_store(z)
    df = z[name]
    print(name + '\n')
    print(f'{df.shape=}, {df.Ticker.nunique()=}\n')
    print(df.iloc[0])


names = [
    'us-derived-banks-quarterly.csv',
    'us-derived-insurance-quarterly.csv',
    'us-derived-quarterly.csv',
    'us-balance-banks-quarterly-full.csv',
    'us-balance-insurance-quarterly-full.csv',
    'us-balance-quarterly-full.csv',
    'us-cashflow-banks-quarterly-full.csv',
    'us-cashflow-insurance-quarterly-full.csv',
    'us-cashflow-quarterly-full.csv',
    'us-income-banks-quarterly-full.csv',
    'us-income-insurance-quarterly-full.csv',
    'us-income-quarterly-full.csv',
]

data_groups = {
    'banks': (
        'us-derived-banks-quarterly.csv',
        'us-balance-banks-quarterly-full.csv',
        'us-cashflow-banks-quarterly-full.csv',
        'us-income-banks-quarterly-full.csv',
    ),
    'insurance': (
        'us-derived-insurance-quarterly.csv',
        'us-balance-insurance-quarterly-full.csv',
        'us-cashflow-insurance-quarterly-full.csv',
        'us-income-insurance-quarterly-full.csv',
    ),
    'rest': (
        'us-derived-quarterly.csv',
        'us-balance-quarterly-full.csv',
        'us-cashflow-quarterly-full.csv',
        'us-income-quarterly-full.csv',
    ),
}


def get_names(names, names_for_group=data_groups):
    if isinstance(names, str):
        group = names
        return names_for_group[group]
    else:
        return names


def get_dfs(dfs, z=None):
    """dfs: iterable of dataframes or names of files for them (found in z)"""
    if z is None:
        z = get_src_store(z)
    if not isinstance(next(iter(dfs)), pd.DataFrame):
        group_or_names = dfs
        names = get_names(group_or_names)
        dfs = tuple(map(z.__getitem__, names))
    return dfs


def common_cols(dfs):
    return tuple(reduce(and_, tuple(map(Pipe(attrgetter('columns'), set), dfs))))


def analyze_group(dfs, z=None):
    z = get_src_store(z)
    dfs = get_dfs(dfs, z)
    d = {}
    d['common_cols'] = common_cols(dfs)
    d['shapes'] = list(map(lambda x: x.shape, dfs))
    merged = pd.concat(dfs, axis=1)
    d['merged_shape'] = merged.shape
    return d


def concat_with_dup_removal(*dfs):
    return remove_duplicate_columns_safely(pd.concat(list(dfs), axis=1))


def duplicated_columns(df):
    return [k for k, v in Counter(df.columns).items() if v > 1]


def remove_last_instance_of_col(df, col):
    last_dup_idx = np.where(df.columns.values == col)[0][-1]
    return df.drop(df.iloc[:, [last_dup_idx]], axis=1)


def remove_duplicate_columns_safely(df):
    df = df.T.drop_duplicates().T
    if dups := duplicated_columns(df):
        print(f'Still some duplicated columns left, will remove last one')
        for col in dups:
            df = remove_last_instance_of_col(df, col)
    return df


def merge_group(dfs, keys=None, z=None):
    z = get_src_store(z)
    dfs = get_dfs(dfs, z)
    return concat_with_dup_removal(*dfs)


def _merge_group_old(dfs, keys=None, z=None):
    z = get_src_store(z)
    dfs = get_dfs(dfs, z)
    if keys is None:
        keys = common_cols(dfs)
    merged = pd.concat(
        dfs, keys=keys, axis=0, ignore_index=True, verify_integrity=True,
    )
    assert len(set(merged.columns)) == len(merged.columns), 'columns are not unique'
    return merged


def complete_with_company_info(d: dict):
    info = get_companies_info()
    ticker = d.pop('Ticker', d.get('ticker', None))
    if ticker is None:
        raise ValueError(f'No ticker found in {d}')
    info_for_ticker = info.T.get(ticker, None)
    if info_for_ticker is not None:
        return dict(info.loc[ticker].to_dict(), **d)
    else:
        return d  # not additional info


def list_diff(lst1, lst2):
    return [x for x in lst1 if x not in lst2]


def group_gather_and_complete_with_info(df, group_cols=('Ticker', 'SimFinId')):
    df = df.dropna(axis=1, how='all')
    if 'Currency' in df.columns:
        if not (df['Currency'] == 'USD').all():
            raise ValueError(f"Currency wasn't all USD!!")
        else:
            df = df.drop(df.loc[:, ['Currency']], axis=1)

    group_cols = list(group_cols)
    dg = df.groupby(group_cols)
    for k, v in dg:
        v = v.dropna(axis=1, how='all')
        t = v.loc[:, list_diff(v.columns, group_cols)]
        yield dict(
            complete_with_company_info(dict(zip(group_cols, k))),
            **t.to_dict(orient='list'),
        )
