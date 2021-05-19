from __future__ import annotations
from typing import Hashable, List, Tuple, Dict, Optional, Sequence, Union, Iterable

import cudf
import pygcylon
from cudf.core.groupby.groupby import _Grouping
from cudf._lib import groupby as cudf_lib_groupby


def _two_lists_equal(l1: List, l2: List) -> bool:
    """
    check whether two lists are equal
    assumes no repeated elements exist
    elements can be in different orders in two lists
    """
    return set(l1) == set(l2)


def _subset_of(subset_list: List, lst: List) -> bool:
    """
    check whether the first list is a subset of the second list
    """
    return all(elem in lst for elem in subset_list)


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

    def _shuffle(self, cdf: cudf.DataFrame = None):
        """
        Shuffle the dataframe if there are multiple workers in the job
        after shuffling, construct the cudf groupby object
        if cdf is None, use the original cudf dataframe
        """
        if cdf is None:
            cdf = self._df.to_cudf()

        if self._env is None or self._env.world_size == 1:
            self._shuffled_cdf = cdf
            self._cudf_groupby = self._shuffled_cdf.groupby(by=self._by,
                                                            level=self._level,
                                                            sort=self._sort,
                                                            as_index=self._as_index,
                                                            dropna=self._dropna)
            return

        shuffle_column_indices = []
        for name in self._grouping_columns:
            if self._level is None:
                shuffle_column_indices.append(cdf._num_indices + cdf._column_names.index(name))
            else:
                shuffle_column_indices.append(cdf._index_names.index(name))

        self._shuffled_cdf = pygcylon.shuffle(cdf, shuffle_column_indices, self._env)
        self._cudf_groupby = self._shuffled_cdf.groupby(by=self._by,
                                                        level=self._level,
                                                        sort=self._sort,
                                                        as_index=self._as_index,
                                                        dropna=self._dropna)


    def _do_groupby(self, key):
        """
        Perform groupby
        this is called from __getattribute__ method
        """

        # if shuffle is not already performed, do it
        if not self._shuffled_cdf:
            self._shuffle()

        result = None
        if self._value_columns:
            result = self._cudf_groupby[self._value_columns].__getattribute__(key)
        else:
            result = self._cudf_groupby.__getattribute__(key)

        return pygcylon.DataFrame.from_cudf(result) if isinstance(result, cudf.DataFrame) else result

    def __getattribute__(self, key):
        """
        if the key is an attribute or this class, return it
        otherwise, if it is a groupby fuction, perform the groupby operation
        """
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
        """
        when a column name or names are given with the subscript operator,
        select those columns as groupby value columns

        if these columns are a subset of the original cudf dataframe,
        create a new dataframe for this subset and perform the shuffle operation on them

        if the shuffle is already performed on those columns in the previous selection,
        there is no need to apply again
        if the shuffle is performed on a superset of columns in the previous selection,
        also there is no need to apply again
        """

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
            raise ValueError("Please provide (int, str, tuple) or list of these as column names: ", key)

        # calculate columns of groupby dataframe,
        # if _level, it is already value_columns,
        # otherwise, it is
        #   groupby columns + value columns
        all_gby_columns = self._value_columns
        if not self._level:
            all_gby_columns = self._grouping_columns + self._value_columns

        # if value_columns are equal to the shuffled_cdf columns, do nothing
        if self._shuffled_cdf and _subset_of(all_gby_columns, self._shuffled_cdf.columns.to_list()):
            return self

        # if the value_columns are equal to all cdf columns, no need to create a subset df
        # shuffle on all columns
        if _two_lists_equal(all_gby_columns, column_names):
            self._shuffle()
            return self

        # create a new dataframe as a subset of columns
        # perform the shuffle on this new cdf
        subset_cdf = self._df.to_cudf()[all_gby_columns]
        self._shuffle(subset_cdf)
        return self
