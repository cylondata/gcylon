def test_mpi():
    from pygcylon import CylonEnv
    from pygcylon.net.mpi_config import MPIConfig
    env: CylonEnv = CylonEnv(config=MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)
    print("finalizing mpi")
    env.finalize()

# test mpi
test_mpi()

class Person:
    def __init__(self, name):
        self._name = name

    @property
    def _myname(self):
        '''name property docs'''
        print('fetch...')
        return self._name

# bob = Person('Bob Smith')
# print(bob._myname)
