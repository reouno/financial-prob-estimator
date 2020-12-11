import inspect
import locale
import os
from datetime import datetime
from typing import List, Union, NamedTuple, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
import quandl
from scipy import stats

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao',
                                   'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# 月や曜日を英語で取得するためこの設定をしておく
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')


class FetchDataByQuandl:
    def __init__(self, api_key: str):
        # quandl.ApiConfig.api_key = "YOUR_KEY_HERE"
        quandl.ApiConfig.api_key = api_key

    def fetch(self, code: str) -> pd.DataFrame:
        df = quandl.get(code)
        df = df[['Last']]
        df = df.rename(columns={'Last': 'Close'})
        validate_df(df)
        return df


def validate_df(df: pd.DataFrame) -> None:
    required_col_names = ['Close']
    required_index_type = pd.DatetimeIndex
    if not isinstance(df.index, required_index_type):
        raise ValueError(f'Type of the data frame index must be {required_index_type}.')
    if not all([col in df.columns for col in required_col_names]):
        raise ValueError(f'All the following columns must exist in the df: {required_col_names}, '
                         f'but the df has only the following cols: {list(df.columns)}')


def to_log_return_ratio_df(df):
    df['logC'] = np.log(df['Close'])
    diff_df = df.diff()
    close_df = df[['Close', 'logC']]
    diff_df = diff_df.rename(columns={'Close': 'CloseDiff', 'logC': 'logCDiff'})
    close_diff_df = diff_df[['CloseDiff', 'logCDiff']]
    close_diff_df['logCDiff'] = close_diff_df['logCDiff'] * 100
    rr_df = pd.concat([close_df, close_diff_df], axis=1)
    rr_df = rr_df.dropna()
    return rr_df


def make_nbars_future(df: pd.DataFrame, bar_range: int, cols: List[str] = None) -> pd.DataFrame:
    """Make n bars dataframe seeing future n bars.
    The row size of `df` must be greater than or equal to `n_bars`, or raise ValueError.
    Args:
        df (DataFrame): target data frame.
        bar_range (int): number of bars.
        cols (List[str], optional): column names. Defaults to ['Close'].
    Raises:
        ValueError: The error is raised when the row size of `df` is smaller than `n_bars`.
    Returns:
        DataFrame: data that can see future n bars
    """
    if not cols:
        cols = ['Close']
    elif not isinstance(cols, list):
        raise ValueError(f'`cols` must be nonempty list, but got {cols}.')

    if df.shape[0] < bar_range:
        raise ValueError(
            f'row size of the df (={df.shape[0]}) must be greater than or equal to '
            f'bar_range (={bar_range + 1})')
    df = df.rename(columns={col: f'{col}0' for col in cols})

    if bar_range == 1:
        return df

    for i in range(1, bar_range):
        for col in cols:
            # FIXME: to_numpy() is too slow!
            # 入力DFのインデックスとしてDateTimeIndexを禁止すればreset_index(drop=True)でいける
            df[f'{col}{i}'] = df[f'{col}0'][i:].append(
                pd.Series([np.nan] * i)).to_numpy()

    df = df.dropna()

    return df


class TParams(NamedTuple):
    df: float
    loc: float  # nearly equal to mean
    scale: float  # nearly equal to std


def estimate_t_params(data: Union[np.ndarray, pd.Series]) -> TParams:
    df, loc, scale = stats.t.fit(data)
    return TParams(df, loc, scale)


def main(data_codes: List[str], bar_range: int, output_csv: Optional[str]=None):
    if not data_codes:
        raise RuntimeError('`data_codes` must not be empty nor None.')

    data_source = FetchDataByQuandl(api_key='Gj-Ed3t3eCzaEgAKeqRH')
    data_list = [data_source.fetch(code) for code in data_codes]
    if len(data_list) > 1:
        d0, subs = data_list
    else:
        d0 = data_list[0]
        subs = []

    # create DF for the analysis
    rr_df = to_log_return_ratio_df(d0)

    # consecutive bars data
    base_df = make_nbars_future(rr_df, bar_range, cols=['logCDiff'])

    # 条件絞り込みロジック
    filtering_condition_col = 'logCDiff0'
    target_col = 'logCDiff1'
    filtering_logic = lambda df: df[df[filtering_condition_col] < 0]

    # 絞り込み結果
    filtered = filtering_logic(base_df)

    t_params_list = [estimate_t_params(df_tmp[target_col]) for df_tmp in [base_df, filtered]]

    if output_csv:
        header = not os.path.exists(output_csv)
        output_df = pd.DataFrame({
            'datetime': [datetime.now()],
            'filtering_condition_col': filtering_condition_col,
            'target_col': target_col,
            'filtering_logic': inspect.getsource(filtering_logic),
            'prior_t_df': t_params_list[0].df,
            'prior_t_loc': t_params_list[0].loc,
            'prior_t_scale': t_params_list[0].scale,
            'posterior_t_df': t_params_list[1].df,
            'posterior_t_loc': t_params_list[1].loc,
            'posterior_t_scale': t_params_list[1].scale,
        })
        output_df.to_csv(output_csv, mode='a', header=header)

    # plot prior and posterior distributions
    # TODO: plot

    print(d0)
    print(subs)
    print(rr_df)
    print(base_df)
    print(filtered)
    print(t_params_list)  # [prior, posterior]


if __name__ == '__main__':
    print('execute estimate-sample-01...')
    main(data_codes=['CHRIS/CME_ES1'], bar_range=2, output_csv='output/result.csv')
