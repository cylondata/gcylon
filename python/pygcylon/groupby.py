from __future__ import annotations
from typing import Hashable, List, Tuple, Dict, Optional, Sequence, Union, Iterable
import pygcylon
from cudf.core.groupby.groupby import _Grouping
from cudf._lib import groupby as cudf_lib_groupby


class GroupByDataFrame(object):

    def __init__(
        self, df: pygcylon.DataFrame, by=None, level=None, sort=False, as_index=True, dropna=True, env: CylonEnv = None
    ):
        self._df = df
        self._by = by
        self._level = level
        self._sort = sort
        self._as_index = as_index
        self._dropna = dropna
        self._env = env

        # determine column names
        if isinstance(by, _Grouping):
            self._grouping_columns = by.names
        else:
            tmp_grouping = _Grouping(df.to_cudf(), by, level)
            self._grouping_columns = tmp_grouping.names

        self._value_columns = []
        self._shuffled_cdf = None
        self._cudf_groupby = None

    def _shuffle(self):
        if self._env is None or self._env.world_size == 1:
            self._shuffled_cdf = self._df.to_cudf()
            return

        shuffle_column_indices = []
        for name in self._grouping_columns:
            if self._level is None:
                shuffle_column_indices.append(self._df._cdf._num_indices + self._df._cdf._column_names.index(name))
            else:
                shuffle_column_indices.append(self._df._cdf._index_names.index(name))

        # todo: if self._value_columns non-empty, create a new dataframe with
        #       self._grouping_columns + self._value_columns and perform the shuffle on them
        self._shuffled_cdf = pygcylon.shuffle(self._df._cdf, shuffle_column_indices, self._env)

    # this is called from __getattribute__ method
    # todo: what should groupby return?
    #       probably a gcylon.DataFrame
    def _do_groupby(self, key):
        # if shuffle is not already performed, do it
        if not self._shuffled_cdf:
            self._shuffle()

        if not self._cudf_groupby:
            self._cudf_groupby = self._shuffled_cdf.groupby(by=self._by,
                                                            level=self._level,
                                                            sort=self._sort,
                                                            as_index=self._as_index,
                                                            dropna=self._dropna)

        if self._value_columns:
            return self._cudf_groupby[self._value_columns].__getattribute__(key)
        else:
            return self._cudf_groupby.__getattribute__(key)

    def __getattribute__(self, key):
        try:
            return super().__getattribute__(key)
        except AttributeError:
            # groupby is performed here
            # if the method is supported, perform groupby
            if key in cudf_lib_groupby._GROUPBY_AGGS:
                return self._do_groupby(key)
            # todo: if key is a column name, cudf.groupby calls groupby on that column
            #   need to check that
            raise

    def __getitem__(self, key):

        # if this method has already been called previously
        prev_value_columns = self._value_columns
        self._value_columns = []

        column_names = self._df._cdf.columns.to_list()
        if isinstance(key, (int, str, tuple)):
            if key in column_names:
                self._value_columns.append(key)
            else:
                raise ValueError("Following column name does not exist in DataFrame: ", key)

        elif isinstance(key, list):
            for aKey in key:
                if aKey not in column_names:
                    raise ValueError("Following column name does not exist in DataFrame: ", key)
            self._value_columns.extend(key)

        else:
            self._value_columns = prev_value_columns
            raise ValueError("Please provide (int, str, tuple) or list of these as column names: ", key)

        # shuffle the dataframe
        # if the dataframe is already shuffled, check whether it is shuffled on the same columns
        # if so, do not reshuffle
        self._shuffle()

        return self

