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

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from cython.operator cimport dereference as deref
from pygcylon.simple cimport testMult
from pygcylon.simple cimport vectorAdd
from pygcylon.simple cimport vectorCopy
from pygcylon.simple cimport Rectangle

from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.table cimport Table

def mult(int n, int m):
   return testMult(n, m)

def listAdd(lst, int m):
    cdef vector[int] vect = lst
    vectorAdd(vect, m)
    return vect

# get shared_ptr from C++
# deref it and return
def listCopy(lst):
    cdef vector[int] vect = lst
    cdef shared_ptr[vector[int]] cp = vectorCopy(vect)
    return deref(cp)

def increment(int n):
    cdef int result = n + 1
    return result

def rows(object tbl):
    cdef Table tb = tbl
    cdef table_view tv = tb.view()
    return tv.num_rows()

def primes(int nb_primes):
    cdef int n, i, len_p
    cdef int p[1000]
    if nb_primes > 1000:
        nb_primes = 1000

    len_p = 0  # The current number of elements in p.
    n = 2
    while len_p < nb_primes:
        # Is n prime?
        for i in p[:len_p]:
            if n % i == 0:
                break

        # If no break occurred in the loop, we have a prime.
        else:
            p[len_p] = n
            len_p += 1
        n += 1

    # Let's return the result in a python list:
    result_as_list  = [prime for prime in p[:len_p]]
    return result_as_list

def primes2(unsigned int nb_primes):
    cdef int n, i
    cdef vector[int] p
    p.reserve(nb_primes)  # allocate memory for 'nb_primes' elements.

    n = 2
    while p.size() < nb_primes:  # size() for vectors is similar to len()
        for i in p:
            if n % i == 0:
                break
        else:
            p.push_back(n)  # push_back is similar to append()
        n += 1

    # Vectors are automatically converted to Python
    # lists when converted to Python objects.
    return p

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class PyRectangle:
    cdef Rectangle c_rect  # Hold a C++ instance which we're wrapping

    def __cinit__(self, int x0, int y0, int x1, int y1):
        self.c_rect = Rectangle(x0, y0, x1, y1)

    def get_area(self):
        return self.c_rect.getArea()

    def get_size(self):
        cdef int width, height
        self.c_rect.getSize(&width, &height)
        return width, height

    def move(self, dx, dy):
        self.c_rect.move(dx, dy)

    cdef get_circum(self):
        cdef int width, height
        self.c_rect.getSize(&width, &height)
        return 2*width + 2*height
