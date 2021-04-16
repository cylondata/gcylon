##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

from libcpp.memory cimport shared_ptr
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from cython.operator cimport dereference as deref
from pygcylon.ctx.context cimport CCylonContext
from pygcylon.ctx.context cimport CylonContext

from pygcylon.data.table cimport Shuffle

from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.column.column cimport column
from cudf._lib.table cimport Table
from cudf._lib.column cimport Column

def shuffle(object tbl, hash_columns, context):
    cdef CStatus status
    cdef Table inputTable = tbl
    cdef table_view inputTview = inputTable.view()
    cdef unique_ptr[table] output
    cdef vector[int] c_hash_columns
    cdef Table outputTable
    cdef CylonContext c_ctx

    if isinstance(context, CylonContext):
        c_ctx = <CylonContext> context
    else:
        raise ValueError('context must be an instance of CylonContext')

    if hash_columns:
        indexColumns = inputTable._num_indices
        hash_columns = [x + indexColumns for x in hash_columns]
        c_hash_columns = hash_columns

        status = Shuffle(inputTview, c_hash_columns, c_ctx.ctx_shd_ptr, output)
        if status.is_ok():
            outputTable = Table.from_unique_ptr(move(output), inputTable._column_names, index_names=inputTable._index_names)
#            outputTable = from_unique_ptr(move(output), inputTable._column_names)
            return outputTable
        else:
            raise ValueError(f"Shuffle operation failed : {status.get_msg().decode()}")
    else:
        raise ValueError('Hash columns are not provided')


cdef Table from_unique_ptr(unique_ptr[table] c_tbl,  object column_names):
        """
        Construct a Table from a unique_ptr to a cudf::table.

        Parameters
        ----------
        c_tbl : unique_ptr[cudf::table]
        column_names : iterable
        """
        cdef vector[unique_ptr[column]] columns
        columns = move(c_tbl.get().release())

        cdef vector[unique_ptr[column]].iterator it = columns.begin()

        # Construct the data OrderedColumnDict
        data_columns = []
#        for i in column_names:
        i = 0
        # ignore the first column
        # not sure how can be an extra column
        it += 1
        while it != columns.end():
            data_columns.append(Column.from_unique_ptr(move(deref(it))))
            it += 1
            i = i + 1

        print("number of columns: ", i)
        data = dict(zip(column_names, data_columns))

        return Table(data=data)
