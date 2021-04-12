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
from pygcylon.ctx.context cimport CCylonContext
from pygcylon.ctx.context cimport CylonContext
from pygcylon.ctx.context import CylonContext
from pygcylon.net.comm_config cimport CCommConfig
from pygcylon.net.mpi_config cimport CMPIConfig
from pygcylon.common.join_config cimport CJoinConfig

from pygcylon.common.status cimport CStatus
from pygcylon.common.status import Status
from pygcylon.common.status cimport Status


cdef api bint pyclon_is_context(object context)

#cdef api shared_ptr[CCommConfig] pycylon_unwrap_comm_config(object comm_config)

cdef api shared_ptr[CCylonContext] pycylon_unwrap_context(object context)

cdef api shared_ptr[CMPIConfig] pycylon_unwrap_mpi_config(object config)

cdef api CJoinConfig* pycylon_unwrap_join_config (object config)

cdef api object pycylon_wrap_context(const shared_ptr[CCylonContext] &ctx)
