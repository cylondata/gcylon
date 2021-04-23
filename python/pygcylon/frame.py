
import cudf
from cudf import _lib as libcudf
from pygcylon.data.table import shuffle
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

    def __init__(self, df):
        if (df is not None) and isinstance(df, cudf.DataFrame):
            self._df = df
        else:
            raise ValueError('A cudf.DataFrame object must be provided.')

    @property
    def df(self) -> cudf.DataFrame:
        return self._df

    def shuffle(self, hash_columns, env: CylonEnv = None) -> cudf.DataFrame:
        tbl = shuffle(self._df, hash_columns, env.context)
        return cudf.DataFrame._from_table(tbl)

    def join(self, other, on=None, how='left', lsuffix='l', rsuffix='r', sort=False, algorithm="hash", env: CylonEnv = None):
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
            return DataFrame(joined_df)

        # shuffle dataframes on index columns
        hash_columns = [*range(self._df._num_indices)]
        shuffled_left = self.shuffle(hash_columns, env)

        hash_columns = [*range(other._df._num_indices)]
        shuffled_right = other.shuffle(hash_columns, env)

        joined_df = shuffled_left.join(shuffled_right,
                                       on=on,
                                       how=how,
                                       lsuffix=lsuffix,
                                       rsuffix=rsuffix,
                                       sort=sort,
                                       method=algorithm)
        return DataFrame(joined_df)

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
              env: CylonEnv = None):
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
            print("merged left: ", merged_df)
            return DataFrame(merged_df)

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
        left_on_ind, right_on_ind = self._get_column_indices(self._df,
                                                             right._df,
                                                             left_on1,
                                                             right_on1,
                                                             left_index,
                                                             right_index)

        shuffled_left = self.shuffle(left_on_ind, env)
        shuffled_right = right.shuffle(right_on_ind, env)

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
        return DataFrame(merged_df)


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
    def _get_column_indices(lhs, rhs, left_on, right_on, left_index, right_index):
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
