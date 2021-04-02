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

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

# from pycylon.cudf.gtable cimport testAdd
#cdef extern from "../../cpp/src/cylon/cudf/gtable.hpp" namespace "gcylon":

# import functions
cdef extern from "../../cpp/src/cylon/cudf/ex.hpp" namespace "gcylon":
    int testMult(int x, int y)

    void vectorAdd(vector[int] &v, int y)

