from __future__ import annotations
from typing import Hashable, List, Dict, Optional, Sequence, Union
import cudf
from pygcylon.data.table import shuffle as tshuffle
from pygcylon.ctx.context import CylonContext


class CylonEnv(object):

    def __init__(self, config=None, distributed=True) -> None:
        self._context = CylonContext(config, distributed)
        self._distributed = distributed
        self._finalized = False

    @property
    def context(self) -> CylonContext:
        return self._context

    @property
    def rank(self) -> int:
        return self._context.get_rank()

    @property
    def world_size(self) -> int:
        return self._context.get_world_size()

    @property
    def is_distributed(self) -> bool:
        return self._distributed

    def finalize(self):
        if not self._finalized:
            self._finalized = True
            self._context.finalize()

    def barrier(self):
        self._context.barrier()

    def __del__(self):
        """
        On destruction of the application, the environment will be automatically finalized
        """
        self.finalize()


class DataFrame(object):

    def __init__(self, data=None, index=None, columns=None, dtype=None):
        """
        A GPU Dataframe object.

        Parameters
        ----------
        data : array-like, Iterable, dict, or DataFrame.
            Dict can contain Series, arrays, constants, or list-like objects.

        index : Index or array-like
            Index to use for resulting frame. Will default to
            RangeIndex if no indexing information part of input data and
            no index provided.

        columns : Index or array-like
            Column labels to use for resulting frame.
            Will default to RangeIndex (0, 1, 2, …, n) if no column
            labels are provided.

        dtype : dtype, default None
            Data type to force. Only a single dtype is allowed.
            If None, infer.
        """
        self._df = cudf.DataFrame(data=data, index=index, columns=columns, dtype=dtype)

    def __repr__(self):
        return self._df.__repr__()

    def __str__(self):
        return self._df.__str__()

    def __setitem__(self, key, value):
        self._df.__setitem__(arg=key, value=value)

    def __getitem__(self, arg):
        return self._df.__getitem__(arg=arg)

    def __setattr__(self, key, col):
        if key == "_df":
            super().__setattr__(key, col) if isinstance(col, cudf.DataFrame) \
                else ValueError("_df has to be an instance of cudf.DataFrame")
        elif self._df:
            self._df.__setattr__(key=key, col=col)
        else:
            raise ValueError("Invalid attribute setting attempt")

    def __getattr__(self, key):
        return self._df if key == "_df" else self._df.__getattr__(key=key)

    def __delitem__(self, name):
        self._df.__delitem__(name=name)

    def __dir__(self):
        return self._df.__dir__()

    def __sizeof__(self):
        return self._df.__sizeof__()

    @staticmethod
    def from_cudf_datafame(cdf) -> DataFrame:
        if (cdf is not None) and isinstance(cdf, cudf.DataFrame):
            df = DataFrame()
            df._df = cdf
            return df
        else:
            raise ValueError('A cudf.DataFrame object must be provided.')


    def join(self,
             other,
             on=None,
             how='left',
             lsuffix='l',
             rsuffix='r',
             sort=False,
             algorithm="hash",
             env: CylonEnv = None) -> DataFrame:
        """Join columns with other DataFrame on index column.

        Parameters
        ----------
        other : DataFrame
        how : str
            Only accepts "left", "right", "inner", "outer"
        lsuffix, rsuffix : str
            The suffices to add to the left (*lsuffix*) and right (*rsuffix*)
            column names when avoiding conflicts.
        sort : bool
            Set to True to ensure sorted ordering.

        Returns
        -------
        joined : DataFrame

        Notes
        -----
        Difference from pandas:

        - *other* must be a single DataFrame for now.
        - *on* is not supported yet due to lack of multi-index support.
        """

        if on is not None:
            raise ValueError('on is not supported with join method. Please use merge method.')

        if env is None:
            joined_df = self._df.join(other=other._df,
                                      on=on,
                                      how=how,
                                      lsuffix=lsuffix,
                                      rsuffix=rsuffix,
                                      sort=sort,
                                      method=algorithm)
            return DataFrame.from_cudf_datafame(joined_df)

        # shuffle dataframes on index columns
        hash_columns = [*range(self._df._num_indices)]
        shuffled_left = shuffle(self._df, hash_columns, env)

        hash_columns = [*range(other._df._num_indices)]
        shuffled_right = shuffle(other._df, hash_columns, env)

        joined_df = shuffled_left.join(shuffled_right,
                                       on=on,
                                       how=how,
                                       lsuffix=lsuffix,
                                       rsuffix=rsuffix,
                                       sort=sort,
                                       method=algorithm)
        return DataFrame.from_cudf_datafame(joined_df)

    def merge(self,
              right,
              how="inner",
              algorithm="hash",
              on=None,
              left_on=None,
              right_on=None,
              left_index=False,
              right_index=False,
              sort=False,
              suffixes=("_x", "_y"),
              copy=True,
              indicator=False,
              validate=None,
              env: CylonEnv = None) -> DataFrame:
        """Merge GPU DataFrame objects by performing a database-style join
        operation by columns or indexes.

        Parameters
        ----------
        right : DataFrame
        on : label or list; defaults to None
            Column or index level names to join on. These must be found in
            both DataFrames.

            If on is None and not merging on indexes then
            this defaults to the intersection of the columns
            in both DataFrames.
        how : {‘left’, ‘outer’, ‘inner’}, default ‘inner’
            Type of merge to be performed.

            - left : use only keys from left frame, similar to a SQL left
              outer join.
            - right : not supported.
            - outer : use union of keys from both frames, similar to a SQL
              full outer join.
            - inner: use intersection of keys from both frames, similar to
              a SQL inner join.
        left_on : label or list, or array-like
            Column or index level names to join on in the left DataFrame.
            Can also be an array or list of arrays of the length of the
            left DataFrame. These arrays are treated as if they are columns.
        right_on : label or list, or array-like
            Column or index level names to join on in the right DataFrame.
            Can also be an array or list of arrays of the length of the
            right DataFrame. These arrays are treated as if they are columns.
        left_index : bool, default False
            Use the index from the left DataFrame as the join key(s).
        right_index : bool, default False
            Use the index from the right DataFrame as the join key.
        sort : bool, default False
            Sort the resulting dataframe by the columns that were merged on,
            starting from the left.
        suffixes: Tuple[str, str], defaults to ('_x', '_y')
            Suffixes applied to overlapping column names on the left and right
            sides
        method : {‘hash’, ‘sort’}, default ‘hash’
            The implementation method to be used for the operation.

        Returns
        -------
            merged : DataFrame

        Notes
        -----
        **DataFrames merges in cuDF result in non-deterministic row ordering.**

        Examples
        --------
        >>> import cudf
        >>> df_a = cudf.DataFrame()
        >>> df_a['key'] = [0, 1, 2, 3, 4]
        >>> df_a['vals_a'] = [float(i + 10) for i in range(5)]
        >>> df_b = cudf.DataFrame()
        >>> df_b['key'] = [1, 2, 4]
        >>> df_b['vals_b'] = [float(i+10) for i in range(3)]
        >>> cdf_a = DataFrame(df_a)
        >>> cdf_b = DataFrame(df_b)
        >>> cdf_merged = cdf_a.merge(cdf_b, on=['key'], how='left')
        >>> cdf_merged.sort_values('key')  # doctest: +SKIP
           key  vals_a  vals_b
        3    0    10.0
        0    1    11.0    10.0
        1    2    12.0    11.0
        4    3    13.0
        2    4    14.0    12.0

        **Merging on categorical variables is only allowed in certain cases**

        Categorical variable typecasting logic depends on both `how`
        and the specifics of the categorical variables to be merged.
        Merging categorical variables when only one side is ordered
        is ambiguous and not allowed. Merging when both categoricals
        are ordered is allowed, but only when the categories are
        exactly equal and have equal ordering, and will result in the
        common dtype.
        When both sides are unordered, the result categorical depends
        on the kind of join:
        - For inner joins, the result will be the intersection of the
        categories
        - For left or right joins, the result will be the the left or
        right dtype respectively. This extends to semi and anti joins.
        - For outer joins, the result will be the union of categories
        from both sides.
        """

        if indicator:
            raise NotImplemented(
                "Only indicator=False is currently supported"
            )

        if env is None:
            merged_df = self._df.merge(right=right._df,
                                       on=on,
                                       left_on=left_on,
                                       right_on=right_on,
                                       left_index=left_index,
                                       right_index=right_index,
                                       how=how,
                                       sort=sort,
                                       suffixes=suffixes,
                                       method=algorithm)
            return DataFrame.from_cudf_datafame(merged_df)

        from cudf.core.join import Merge
        # just for checking purposes, we assign "left" to how if it is "right"
        howToCheck = "left" if how == "right" else how
        Merge.validate_merge_cfg(lhs=self._df,
                                 rhs=right._df,
                                 on=on,
                                 left_on=left_on,
                                 right_on=right_on,
                                 left_index=left_index,
                                 right_index=right_index,
                                 how=howToCheck,
                                 lsuffix=suffixes[0],
                                 rsuffix=suffixes[1],
                                 suffixes=suffixes)

        left_on1, right_on1 = self._get_left_right_on(self._df,
                                                      right._df,
                                                      on,
                                                      left_on,
                                                      right_on,
                                                      left_index,
                                                      right_index)
        left_on_ind, right_on_ind = DataFrame._get_left_right_indices(self._df,
                                                                      right._df,
                                                                      left_on1,
                                                                      right_on1,
                                                                      left_index,
                                                                      right_index)

        shuffled_left = shuffle(self._df, left_on_ind, env)
        shuffled_right = shuffle(right._df, right_on_ind, env)

        merged_df = shuffled_left.merge(right=shuffled_right,
                                        on=on,
                                        left_on=left_on,
                                        right_on=right_on,
                                        left_index=left_index,
                                        right_index=right_index,
                                        how=how,
                                        sort=sort,
                                        suffixes=suffixes,
                                        method=algorithm)
        return DataFrame.from_cudf_datafame(merged_df)

    @staticmethod
    def _get_left_right_on(lhs, rhs, on, left_on, right_on, left_index, right_index):
        """
        Calculate left_on and right_on as a list of strings (column names)
        this is based on the "preprocess_merge_params" function in the cudf file:
            cudf/core/join/join.py
        """
        if on:
            on = [on] if isinstance(on, str) else list(on)
            left_on = right_on = on
        else:
            if left_on:
                left_on = (
                    [left_on] if isinstance(left_on, str) else list(left_on)
                )
            if right_on:
                right_on = (
                    [right_on] if isinstance(right_on, str) else list(right_on)
                )

        same_named_columns = [value for value in lhs._column_names if value in rhs._column_names]
        if not (left_on or right_on) and not (left_index and right_index):
            left_on = right_on = list(same_named_columns)

        return left_on, right_on

    @staticmethod
    def _get_left_right_indices(lhs, rhs, left_on, right_on, left_index, right_index):
        """
        Calculate left and right column indices to perform shuffle on
        this is based on the "join" function in cudf file:
            cudf/_lib/join.pyx
        """

        if left_on is None:
            left_on = []
        if right_on is None:
            right_on = []

        left_on_ind = []
        right_on_ind = []

        if left_index or right_index:
            # If either true, we need to process both indices as columns
            left_join_cols = list(lhs._index_names) + list(lhs._column_names)
            right_join_cols = list(rhs._index_names) + list(rhs._column_names)

            if left_index and right_index:
                # Both dataframes must take index column indices
                left_on_indices = right_on_indices = range(lhs._num_indices)

            elif left_index:
                # Joins left index columns with right 'on' columns
                left_on_indices = range(lhs._num_indices)
                right_on_indices = [
                    right_join_cols.index(on_col) for on_col in right_on
                ]

            elif right_index:
                # Joins right index columns with left 'on' columns
                right_on_indices = range(rhs._num_indices)
                left_on_indices = [
                    left_join_cols.index(on_col) for on_col in left_on
                ]

            for i_l, i_r in zip(left_on_indices, right_on_indices):
                left_on_ind.append(i_l)
                right_on_ind.append(i_r)

        else:
            left_join_cols = list(lhs._index_names) + list(lhs._column_names)
            right_join_cols = list(rhs._index_names) + list(rhs._column_names)

        # If both left/right_index, joining on indices plus additional cols
        # If neither, joining on just cols, not indices
        # In both cases, must match up additional column indices in lhs/rhs
        if left_index == right_index:
            for name in left_on:
                left_on_ind.append(left_join_cols.index(name))
            for name in right_on:
                right_on_ind.append(right_join_cols.index(name))

        return left_on_ind, right_on_ind

    def _get_column_indices(self) -> List[int]:
        """
        Get the column indices excluding index columns
        :return: list of ints
        """
        lists = DataFrame._get_all_column_indices([self])
        return lists[0]

    @staticmethod
    def _get_all_column_indices(dfs) -> List[List[int]]:
        """
        Get indices of all DataFrames excluding index columns
        This is to calculate indices of columns that will be used
        to perform partitioning/shuffling on the dataframe
        :param dfs: list of DataFrame objects
        :return: list of list of column indices
        """
        all_df_indices = [];
        for cdf in dfs:
            df_indices = [*range(cdf._df._num_indices, cdf._df._num_indices + cdf._df._num_columns)]
            all_df_indices.append(df_indices)
        return all_df_indices

    @staticmethod
    def _get_all_common_indices(dfs) -> List[List[int]]:
        """
        Get indices of all columns common in all DataFrames
        Columns might be in different indices in different DataFrames
        This is to calculate indices of columns that will be used
        to perform partitioning/shuffling on the dataframe
        :param dfs: list of DataFrame objects
        :return: list of list of column indices
        """

        # get the inersection of all column names
        common_columns_names = DataFrame._get_common_column_names(dfs)
        if len(common_columns_names) == 0:
            raise ValueError("There is no common column names among the provided DataFrame objects")

        all_df_indices = [];
        for cdf in dfs:
            df_indices = []
            col_names = list(cdf._df._index_names) + list(cdf._df._column_names)
            for name in common_columns_names:
                df_indices.append(col_names.index(name))
            all_df_indices.append(df_indices)
        return all_df_indices

    @staticmethod
    def _get_common_column_names(dfs) -> List[str]:
        """
        Get common column names in the proved DataFrames
        :param dfs: list of DataFrame objects
        :return: list of column names that are common to all DataFrames
        """
        column_name_lists = [list(obj._df._column_names) for obj in dfs]
        common_column_names = set(column_name_lists[0])
        for column_names in column_name_lists[1:]:
            common_column_names = common_column_names & set(column_names)
        return common_column_names

    def drop_duplicates(
            self,
            subset: Optional[Union[Hashable, Sequence[Hashable]]] = None,
            keep: Union[str, bool] = "first",
            inplace: bool = False,
            ignore_index: bool = False,
            env: CylonEnv = None) -> Union[DataFrame or None]:
        """
        Remove duplicate rows from the DataFrame.
        Considering certain columns is optional. Indexes, including time indexes
        are ignored.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', False}, default 'first'
            Determines which duplicates (if any) to keep.
            - ``first`` : Drop duplicates except for the first occurrence.
            - ``last`` : Drop duplicates except for the last occurrence.
            - False: Drop all duplicates.
        inplace : bool, default False
            Whether to drop duplicates in place or to return a copy.
            inplace is supported only in local mode
            when there are multiple workers in the computation, inplace is disabled
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        env: CylonEnv object

        Returns
        -------
        DataFrame or None
            DataFrame with duplicates removed or
            None if ``inplace=True`` and in the local mode with no distributed workers.
        """

        if env is None or env.world_size == 1:
            dropped_df = self._df.drop_duplicates(subset=subset, keep=keep, inplace=inplace, ignore_index=ignore_index)
            return DataFrame.from_cudf_datafame(dropped_df) if not inplace else None

        shuffle_column_indices = []
        if subset is None:
            shuffle_column_indices = self._get_column_indices()
        elif isinstance(subset, str):
            shuffle_column_indices.append(self._df._num_indices + self._df._column_names.index(subset))
        elif len(subset) == 0:
            raise ValueError("subset is empty. it should be either None or sequence of column names")
        else:
            for name in subset:
                shuffle_column_indices.append(self._df._num_indices + self._df._column_names.index(name))

        shuffled_df = shuffle(self._df, shuffle_column_indices, env)

        dropped_df = shuffled_df.drop_duplicates(subset=subset, keep=keep, inplace=inplace, ignore_index=ignore_index)
        return DataFrame.from_cudf_datafame(dropped_df) if dropped_df else DataFrame.from_cudf_datafame(shuffled_df)

    def set_index(
            self,
            keys,
            drop=True,
            append=False,
            inplace=False,
            verify_integrity=False,
    ) -> Union[DataFrame or None]:
        """Return a new DataFrame with a new index

        Parameters
        ----------
        keys : Index, Series-convertible, label-like, or list
            Index : the new index.
            Series-convertible : values for the new index.
            Label-like : Label of column to be used as index.
            List : List of items from above.
        drop : boolean, default True
            Whether to drop corresponding column for str index argument
        append : boolean, default True
            Whether to append columns to the existing index,
            resulting in a MultiIndex.
        inplace : boolean, default False
            Modify the DataFrame in place (do not create a new object).
        verify_integrity : boolean, default False
            Check for duplicates in the new index.

        Returns
        -------
        DataFrame or None
            DataFrame with a new index or
            None if ``inplace=True``

        Examples
        --------
        >>> df = cudf.DataFrame({
        ...     "a": [1, 2, 3, 4, 5],
        ...     "b": ["a", "b", "c", "d","e"],
        ...     "c": [1.0, 2.0, 3.0, 4.0, 5.0]
        ... })
        >>> df
           a  b    c
        0  1  a  1.0
        1  2  b  2.0
        2  3  c  3.0
        3  4  d  4.0
        4  5  e  5.0

        Set the index to become the ‘b’ column:

        >>> df.set_index('b')
           a    c
        b
        a  1  1.0
        b  2  2.0
        c  3  3.0
        d  4  4.0
        e  5  5.0

        Create a MultiIndex using columns ‘a’ and ‘b’:

        >>> df.set_index(["a", "b"])
               c
        a b
        1 a  1.0
        2 b  2.0
        3 c  3.0
        4 d  4.0
        5 e  5.0

        Set new Index instance as index:

        >>> df.set_index(cudf.RangeIndex(10, 15))
            a  b    c
        10  1  a  1.0
        11  2  b  2.0
        12  3  c  3.0
        13  4  d  4.0
        14  5  e  5.0

        Setting `append=True` will combine current index with column `a`:

        >>> df.set_index("a", append=True)
             b    c
          a
        0 1  a  1.0
        1 2  b  2.0
        2 3  c  3.0
        3 4  d  4.0
        4 5  e  5.0

        `set_index` supports `inplace` parameter too:

        >>> df.set_index("a", inplace=True)
        >>> df
           b    c
        a
        1  a  1.0
        2  b  2.0
        3  c  3.0
        4  d  4.0
        5  e  5.0
        """

        indexed_df = self._df.set_index(keys=keys, drop=drop, append=append, inplace=inplace,
                                       verify_integrity=verify_integrity)
        return DataFrame.from_cudf_datafame(indexed_df) if indexed_df else None

    def reset_index(
            self, level=None, drop=False, inplace=False, col_level=0, col_fill=""
    ) -> Union[DataFrame or None]:
        """
        Reset the index.

        Reset the index of the DataFrame, and use the default one instead.

        Parameters
        ----------
        drop : bool, default False
            Do not try to insert index into dataframe columns. This resets
            the index to the default integer index.
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).

        Returns
        -------
        DataFrame or None
            DataFrame with the new index or None if ``inplace=True``.

        Examples
        --------
        >>> df = cudf.DataFrame([('bird', 389.0),
        ...                    ('bird', 24.0),
        ...                    ('mammal', 80.5),
        ...                    ('mammal', np.nan)],
        ...                   index=['falcon', 'parrot', 'lion', 'monkey'],
        ...                   columns=('class', 'max_speed'))
        >>> df
                 class max_speed
        falcon    bird     389.0
        parrot    bird      24.0
        lion    mammal      80.5
        monkey  mammal      <NA>
        >>> df.reset_index()
            index   class max_speed
        0  falcon    bird     389.0
        1  parrot    bird      24.0
        2    lion  mammal      80.5
        3  monkey  mammal      <NA>
        >>> df.reset_index(drop=True)
            class max_speed
        0    bird     389.0
        1    bird      24.0
        2  mammal      80.5
        3  mammal      <NA>
        """
        indexed_df = self._df.reset_index(level=level, drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill)
        return DataFrame.from_cudf_datafame(indexed_df) if indexed_df else None


def concat(
        dfs,
        axis=0,
        join="outer",
        ignore_index: bool = False,
        keys=None,
        levels=None,
        names=None,
        verify_integrity: bool = False,
        sort: bool = False,
        copy: bool = True,
        env: CylonEnv = None
) -> DataFrame:
    """Concatenate DataFrames row-wise.

    Parameters
    ----------
    dfs: list of DataFrames to concatenate
    axis: {0/'index', 1/'columns'}, default 0
        The axis to concatenate along. (Currently only 0 supported)
    join: {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis (or axes).
    ignore_index: bool, default False
        Set True to ignore the index of the *dfs* and provide a
        default range index instead.
    keys (Unsupported) : sequence, default None
        If multiple levels passed, should contain tuples. Construct
        hierarchical index using the passed keys as the outermost level.
    levels (Unsupported) : list of sequences, default None
        Specific levels (unique values) to use for constructing a
        MultiIndex. Otherwise they will be inferred from the keys.
    names (Unsupported) : list, default None
        Names for the levels in the resulting hierarchical index.
    verify_integrity (Unsupported) : bool, default False
        Check whether the new concatenated axis contains duplicates. This can
        be very expensive relative to the actual data concatenation.
    sort : bool, default False
        Sort non-concatenation axis if it is not already aligned when `join`
        is 'outer'.
        This has no effect when ``join='inner'``, which already preserves
        the order of the non-concatenation axis.
    copy (Unsupported) : bool, default True
        If False, do not copy data unnecessarily.
    env: Cylon environment object

    Returns
    -------
    A new DataFrame object constructed by concatenating all input DataFrame objects.

    Examples
    --------

    Combine two ``DataFrame`` objects with identical columns.

    >>> df1 = DataFrame([['a', 1], ['b', 2]],
    ...                    columns=['letter', 'number'])
    >>> df1
    letter  number
    0      a       1
    1      b       2
    >>> df2 = DataFrame([['c', 3], ['d', 4]],
    ...                    columns=['letter', 'number'])
    >>> df2
    letter  number
    0      c       3
    1      d       4
    >>> DataFrame.concat([df1, df2])
    letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects with overlapping columns
    and return everything. Columns outside the intersection will
    be filled with ``NaN`` values.

    >>> df3 = DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],
    ...                    columns=['letter', 'number', 'animal'])
    >>> df3
    letter  number animal
    0      c       3    cat
    1      d       4    dog
    >>> DataFrame.concat([df1, df3], sort=False)
    letter  number animal
    0      a       1    NaN
    1      b       2    NaN
    0      c       3    cat
    1      d       4    dog

    Combine ``DataFrame`` objects with overlapping columns
    and return only those that are shared by passing ``inner`` to
    the ``join`` keyword argument.

    >>> DataFrame.concat([df1, df3], join="inner")
    letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    (Unsupported) Combine ``DataFrame`` objects horizontally along the x axis by
    passing in ``axis=1``.

    >>> df4 = DataFrame([['bird', 'polly'], ['monkey', 'george']],
    ...                    columns=['animal', 'name'])
    >>> DataFrame.concat([df1, df4], axis=1)

    letter  number  animal    name
    0      a       1    bird   polly
    1      b       2  monkey  george

    (Unsupported) Prevent the result from including duplicate index values with the
    ``verify_integrity`` option.

    >>> df5 = DataFrame([1], index=['a'])
    >>> df5
    0
    a  1
    >>> df6 = DataFrame([2], index=['a'])
    >>> df6
    0
    a  2
    >>> DataFrame.concat([df5, df6], verify_integrity=True)
    Traceback (most recent call last):
        ...
    ValueError: Indexes have overlapping values: ['a']
    """

    if not dfs:
        raise ValueError("No DataFrames to concatenate")

    # remove None objects if any
    dfs = [obj for obj in dfs if obj is not None]
    if len(dfs) == 0:
        raise ValueError("No DataFrames to concatenate after None removal")

    if axis != 0:
        raise ValueError("Only concatenation on axis 0 is currently supported")

    if verify_integrity not in (None, False):
        raise NotImplementedError("verify_integrity parameter is not supported yet.")

    if keys is not None:
        raise NotImplementedError("keys parameter is not supported yet.")

    if levels is not None:
        raise NotImplementedError("levels parameter is not supported yet.")

    if names is not None:
        raise NotImplementedError("names parameter is not supported yet.")

    if not copy:
        raise NotImplementedError("copy can be only True.")

    # make sure all dfs DataFrame objects
    for obj in dfs:
        if not isinstance(obj, DataFrame):
            raise ValueError("Only DataFrame objects can be concatenated")

    # perform local concatenation, no need to distributed concat
    dfs = [obj._df for obj in dfs]
    concated_df = cudf.concat(dfs, axis=axis, join=join, ignore_index=ignore_index, sort=sort)
    return DataFrame.from_cudf_datafame(concated_df)


def shuffle(df: cudf.DataFrame, hash_columns, env: CylonEnv = None) -> cudf.DataFrame:
    """
    Perform shuffle on a distributed dataframe
    :param df: local DataFrame object
    :param hash_columns: column indices to partition the table
    :param env: CylonEnv
    :return: shuffled dataframe as a new object
    """
    tbl = tshuffle(df, hash_columns, env.context)
    return cudf.DataFrame._from_table(tbl)
