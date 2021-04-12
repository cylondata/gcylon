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

from pygcylon.ctx.context cimport CCylonContext
from pygcylon.ctx.context cimport CylonContext
from pygcylon.ctx.context import CylonContext
from pygcylon.net.comm_config cimport CCommConfig
from pygcylon.net.mpi_config cimport CMPIConfig
from pygcylon.net.mpi_config import MPIConfig
from pygcylon.net.mpi_config cimport MPIConfig
from pygcylon.common.join_config cimport CJoinConfig
from pygcylon.common.join_config import JoinConfig
from pygcylon.common.join_config cimport JoinConfig
from pygcylon.common.status cimport CStatus
from pygcylon.common.status import Status
from pygcylon.common.status cimport Status

cdef api bint pyclon_is_context(object context):
    return isinstance(context, CylonContext)

cdef api bint pyclon_is_mpi_config(object mpi_config):
    return isinstance(mpi_config, MPIConfig)

cdef api bint pyclon_is_join_config(object config):
    return isinstance(config, JoinConfig)

cdef api shared_ptr[CCylonContext] pycylon_unwrap_context(object context):
    cdef CylonContext ctx
    if pyclon_is_context(context):
        ctx = <CylonContext> context
        return ctx.ctx_shd_ptr
    return CCylonContext.Init()

cdef api shared_ptr[CMPIConfig] pycylon_unwrap_mpi_config(object config):
    cdef MPIConfig mpi_config
    if pyclon_is_mpi_config(config):
        mpi_config = <MPIConfig> config
        return mpi_config.mpi_config_shd_ptr
    else:
        raise ValueError('Passed object is not an instance of MPIConfig')

cdef api CJoinConfig* pycylon_unwrap_join_config (object config):
    cdef JoinConfig jc
    if pyclon_is_join_config(config):
        jc = <JoinConfig> config
        return jc.jcPtr
    else:
        raise ValueError('Passed object is not an instance of JoinConfig')

cdef api object pycylon_wrap_context(const shared_ptr[CCylonContext] &ctx):
    cdef CylonContext context = CylonContext.__new__(CylonContext)
    context.init(ctx)
    return context
