"""Functions to prepare company info jsons"""
# ---------------------------------------------------------------------------------------
# util
from functools import partial
import numpy as np
from functools import lru_cache

def filtered_list(condition, iterable):
    from typing import Container

    if not callable(condition):
        if isinstance(condition, Container):
            condition = condition.__contains__
        else:
            raise TypeError(f'Unrecognized condition type: {type(condition)}')
    return list(filter(condition, iterable))


def move_column_inplace(df, colname, position_idx=0):
    col = df[colname]
    df.drop(labels=[colname], axis=1, inplace=True)
    df.insert(position_idx, colname, col)


import requests
from dol import Pipe
from operator import attrgetter
from graze import graze

content_url = 'https://raw.githubusercontent.com/thorwhalen/content/master/{}'.format
get_tw_content = Pipe(content_url, requests.get, attrgetter('content'))
graze_tw_content = Pipe(content_url, graze)

# ---------------------------------------------------------------------------------------
# preparing company infos


import pandas as pd


def prepare_info_df(df, rename_cols: dict = None):
    if rename_cols:
        df = df.rename(columns=rename_cols)
    move_column_inplace(df, 'ticker', 0)
    df = df.set_index('ticker', drop=False)
    df.index.name = None
    return df


@lru_cache
def get_companies_info():
    from io import BytesIO
    import pandas as pd
    from dol import FilesOfZip

    z = FilesOfZip('local/companies info.zip')
    df = pd.read_csv(BytesIO(z['companies info/companies.csv']))

    df = prepare_info_df(df, rename_cols={'Ticker': 'ticker'})

    df['tags'] = df.apply(
        lambda row: row[['tag 1', 'tag 2', 'tag 3']].dropna().to_list() or float('nan'),
        axis=1,
    )
    del df['tag 1']
    del df['tag 2']
    del df['tag 3']

    assert not any(df['ticker'].isna())
    assert len(df['ticker'].unique()) == len(df)
    #     df = df.set_index('ticker')
    return df


def nr_schema_records(df):
    for idx, row in df.iterrows():
        row = row.dropna().to_dict()
        if 'id' in row and isinstance(row['id'], float) and row['id'] == int(row['id']):
            row['id'] = int(row['id'])
        yield dict(ticker=idx, **row)


def save_nr_schema_records_to_json(df, filepath=None):
    import json

    filepath = filepath or 'company_info.json'
    if not filepath.endswith('.json'):
        filepath += '.json'
    with open(filepath, 'wt') as fp:
        json.dump(list(nr_schema_records(df)), fp)
    return filepath


# ---------------------------------------------------------------------------------------
# dataframe to dict


def only_ints_or_nan(arr, is_missing=np.isnan):
    """Determines if all non-missing values are 'effectively' ints (e.g. 7.0) """
    try:
        not_missing_lidx = ~is_missing(arr)
        return all(arr[not_missing_lidx].astype(int) == arr[not_missing_lidx])
    except TypeError:
        return False


def only_ints_of_nan_columns(df, is_missing=np.isnan):
    return list(
        map(
            lambda kv: kv[0],
            filter(lambda kv: only_ints_or_nan(kv[1], is_missing), df.items()),
        )
    )


def rm_rows_and_cols_with_all_nans(df):
    df = df.copy()
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    return df


def row_to_light_dict(row, **key_transforms):
    d = row.dropna().to_dict()
    for k in d.keys() & key_transforms.keys():
        d[k] = key_transforms[k](d[k])  # apply the transform for key k
    return d


def df_to_light_records(df):
    df = rm_rows_and_cols_with_all_nans(df)
    int_columns = only_ints_of_nan_columns(df)
    # key_transforms = {'int_col_1': int, 'int_col_2': int,...}
    key_transforms = dict(zip(int_columns, [int] * len(int_columns)))
    for ticker, row in df.iterrows():
        yield ticker, row_to_light_dict(row, **key_transforms)


# ---------------------------------------------------------------------------------------
# yahoo source


def get_yahoo_companies_info_from_mongo():
    from mongodol import MongoClientReader
    from dol import wrap_kvs

    df = pd.DataFrame(MongoClientReader()['yf']['info'].values())
    assert all(df['symbol'] == df['_id'])
    del df['_id']
    df = prepare_info_df(df, rename_cols={'symbol': 'ticker'})
    return df


def mk_light_yahoo_companies_info_json(
    yahoo_df=None, filepath='companies_info_from_yahoo'
):
    from dol import to_zipped_bytes
    from pathlib import Path

    if yahoo_df is None:
        yahoo_df = get_yahoo_companies_info_from_mongo

    d = df_to_light_records(yahoo_df)
    if filepath.endswith('.zip'):
        Path(filepath).write_bytes(to_zipped_bytes(json.dumps(d).encode()))
    else:
        Path(filepath).write_text(json.dumps(d))


import json
from graze import graze
from dol import zipped_bytes_to_bytes, Pipe

get_yahoo_companies_info = Pipe(
    partial(content_url, 'json/companies_info.json.zip'),  # make the src url
    graze,  # get the contents for url (using local cache)
    zipped_bytes_to_bytes,  # convert zip bytes to unzipped bytes
    json.loads,  # unjasonize the bytes into a list of dicts
    pd.DataFrame,  # convert to dataframe
    pd.DataFrame.transpose  # transform
)

# def get_yahoo_companies_info():
#
#     return pd.DataFrame(
#         json.loads(
#             zipped_bytes_to_bytes(graze(content_url('json/companies_info.json.zip')))
#         )
#     ).T


# ---------------------------------------------------------------------------------------
# Visualization


def visualize_missing_data(
    df,
    *,
    is_missing=pd.isna,
    field_order='most_complete',
    figsize=(12, 12),
    heatmap_kwargs=(),
):
    """Make a heatmap of missing values of a data frame.

    """
    import matplotlib.pylab as plt
    from seaborn import heatmap

    plt.figure(figsize=figsize)
    t = -is_missing(df)
    if field_order is not None:
        if isinstance(field_order, str):
            if field_order == 'most_complete':
                field_order = t.sum().sort_values(ascending=False).index.values.tolist()
            else:
                raise ValueError(f'Unrecognized field_order: {field_order}')
        t = t.sort_values(by=field_order, ascending=False)
        t = t[field_order]
    heatmap_kwargs = dict(cbar=False, **dict(heatmap_kwargs))
    return heatmap(t, **heatmap_kwargs)
