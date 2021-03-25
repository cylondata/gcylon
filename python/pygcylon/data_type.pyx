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


#from pycylon.cudf.gtable cimport testAdd
from libcpp.vector cimport vector


def serve():
    result = testAdd(10, 20)


#def primes(unsigned int nb_primes):
#    cdef int n, i
#    cdef vector[int] p
#    p.reserve(nb_primes)  # allocate memory for 'nb_primes' elements.

#    n = 2
#    while p.size() < nb_primes:  # size() for vectors is similar to len()
#        for i in p:
#            if n % i == 0:
#                break
#        else:
#            p.push_back(n)  # push_back is similar to append()
#        n += 1

    # Vectors are automatically converted to Python
    # lists when converted to Python objects.
#    return p