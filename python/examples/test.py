def test_mpi():
    import pygcylon
    env: pygcylon.CylonEnv = pygcylon.CylonEnv(config=pygcylon.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)
    print("finalizing mpi")
    env.finalize()

# test mpi
test_mpi()
