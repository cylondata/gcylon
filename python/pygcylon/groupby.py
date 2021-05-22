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
        """
        Group DataFrame using a mapper or by a Series of columns.
        Works with both single DataFrame or distributed DataFrame

        A groupby operation involves some combination of splitting the object,
        applying a function, and combining the results. This can be used to
        group large amounts of data and compute operations on these groups.

        When calculating groupby in distributed DataFrames,
        an all-to-all shuffle communication operation is performed.
        It is very important to avoid unnecessary shuffle operations,
        since the shuffling of the dataframe among distributed workers are constly.

        When a GroupByDataFrame object is created, and the first groupby operation is performed,
        this shuffle operation is performed by partitioning the tables on the groupby columns and
        all dataframe columns are shuffled.

        So, to get the best performance in a distributed dataframe,
        one should first create a GroupByDataFrame object and perform many aggregations on it.
        Because, creating and performing a groupby object requires a distributed shuffle.
        When we reuse the same GroupByDataFrame object, we avoid re-shuffling the dataframe.
        For example following code performs a single shuffle only:
            gby = df.groupby(["column1", "column2"], ..., env=env)
            gby.sum()
            gby["columnx"].mean()
            gby[["columnx", "columny"]].min()

        One must avoid running the groupby operation on the dataframe object.
        For example, all three of the following operations perform the a separate distributed shuffle:
            df.groupby("columnq", env=env)["columnb"].sum()
            df.groupby("columnq", env=env)["columnb"].mean()
            df.groupby("columnq", env=env)["columnc"].max()
        One can easily perform a single shuffle for these three lines by first creating a GroupByDataFrame object
        and performing the aggragations using it.

        A second important point is to create a new dataframe from a subset of columns
        and performing the groupby on it when working with dataframes with many columns.
        Suppose, you are working with a dataframe with hundreds of columns
        but you would like to perform the groupby and aggregations on a small number of columns.
        First, you need to create a new dataframe with those groupby and aggregations columns.
        Then, perform the groupby on this new dataframe.
        This will avoid shufling the whole dataframe. Only the columns on the new dataframe will be shuffled.
            df2 = df[["columnx", "columny", "columnz", ...]]
            gby = df2.groupby("columnx", env=env)
            gby["columny"].sum()
            gby["columnz"].mean()
        In this case, the shuffling is performed only on the columns of df2.


        Parameters
        ----------
        by : mapping, function, label, or list of labels
            Used to determine the groups for the groupby. If by is a
            function, it’s called on each value of the object’s index.
            If a dict or Series is passed, the Series or dict VALUES will
            be used to determine the groups (the Series’ values are first
            aligned; see .align() method). If a cupy array is passed, the
            values are used as-is determine the groups. A label or list
            of labels may be passed to group by the columns in self.
            Notice that a tuple is interpreted as a (single) key.
        level : int, level name, or sequence of such, default None
            If the axis is a MultiIndex (hierarchical), group by a particular
            level or levels.
        as_index : bool, default True
            For aggregated output, return object with group labels as
            the index. Only relevant for DataFrame input.
            as_index=False is effectively “SQL-style” grouped output.
        sort : bool, default False
            Sort result by group key. Differ from Pandas, cudf defaults to
            ``False`` for better performance. Note this does not influence
            the order of observations within each group. Groupby preserves
            the order of rows within each group.
        dropna : bool, optional
            If True (default), do not include the "null" group.
        env: CylonEnv needs to be provided for distributed groupby operation.

        Returns
        -------
            DataFrameGroupBy
                Returns a groupby object that contains information
                about the groups.

        Examples
        --------
        >>> import pygcylon as gc
        >>> # first try local groupby on a single DataFrame
        >>> env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
        >>> df = gc.DataFrame({'a': [1, 1, 1, 2, 2], 'b': [1, 1, 2, 2, 3], 'c': [1, 2, 3, 4, 5]})
        >>> df
           a  b  c
        0  1  1  1
        1  1  1  2
        2  1  2  3
        3  2  2  4
        4  2  3  5
        >>> # create a groupby object and perform multiple operations
        >>> gby = df.groupby("a")
        >>> gby.sum()
           b  c
        a
        2  5  9
        1  4  6
        >>> gby.max()
           b  c
        a
        2  3  5
        1  2  3
        >>> gby["b"].sum()
           b
        a
        2  5
        1  4
        >>> # to perform groupby on a different set of columns, we need to create a new GroupByDataFrame object
        >>> gby = df.groupby(["a", "b"])
        >>> gby.sum()
             c
        a b
        1 2  3
        2 2  4
          3  5
        1 1  3

        >>> # todo: add distributed DataFrame examples
        >>> env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)

        """

        self._df = df
        self._by = by
        self._level = level
        self._sort = sort
        self._as_index = as_index
        self._dropna = dropna
        self._env = env

        # initialize:
        #   self._grouping_columns
        if isinstance(by, _Grouping):
            self._grouping_columns = by.names
        else:
            tmp_grouping = _Grouping(df.to_cudf(), by, level)
            self._grouping_columns = tmp_grouping.names

        # shuffle the dataframe
        # initialize:
        #   self._shuffled_cdf
        #   self._cudf_groupby
        self._shuffle()
        self._value_columns = []

    def _shuffle(self):
        """
        Shuffle the dataframe if there are multiple workers in the job
        after shuffling, construct the cudf groupby object
        if cdf is None, use the original cudf dataframe
        """
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

        if self._value_columns:
            try:
                result = self._cudf_groupby[self._value_columns].__getattribute__(key)
            finally:
                self._value_columns = []
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
        Given column names are saved to e used in the next call of __getattribute__ method
        when the next groupby operation is performed After the call of __getattribute__ method,
        these column names are deleted.

        We assume that after this method, an aggregation function will always be called.
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

        return self

    def __iter__(self):
        """
        Iterating through group-name & grouped values
        """
        return self._cudf_groupby.__iter__()

    def size(self):
        """
        Return the size of each group.
        """
        return self._cudf_groupby.size()

    @property
    def groups(self):
        """
        Returns a dictionary mapping group keys to row labels.
        """
        return self._cudf_groupby.groups

    def agg(self, func):
        """
        Apply aggregation(s) to the groups.

        Parameters
        ----------
        func : str, callable, list or dict

        Returns
        -------
        A Series or DataFrame containing the combined results of the
        aggregation.

        Examples
        --------
        >>> import pygcylon as gc
        >>> df = gc.DataFrame(
            {'a': [1, 1, 2], 'b': [1, 2, 3], 'c': [2, 2, 1]})
        >>> gby = df.groupby('a')
        >>> gby.agg('sum')
           b  c
        a
        2  3  1
        1  3  4

        Specifying a list of aggregations to perform on each column.

        >>> gby.agg(['sum', 'min'])
            b       c
          sum min sum min
        a
        2   3   3   1   1
        1   3   1   4   2

        Using a dict to specify aggregations to perform per column.

        >>> gby.agg({'a': 'max', 'b': ['min', 'mean']})
            a   b
          max min mean
        a
        2   2   3  3.0
        1   1   1  1.5

        Using lambdas/callables to specify aggregations taking parameters.

        >>> f1 = lambda x: x.quantile(0.5); f1.__name__ = "q0.5"
        >>> f2 = lambda x: x.quantile(0.75); f2.__name__ = "q0.75"
        >>> gby.agg([f1, f2])
             b          c
          q0.5 q0.75 q0.5 q0.75
        a
        1  1.5  1.75  2.0   2.0
        2  3.0  3.00  1.0   1.0
        """
        return self._cudf_groupby.agg(func=func)

    def nth(self, n):
        """
        Return the nth row from each group.
        """
        return self._cudf_groupby.nth(n=n)
