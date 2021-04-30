import cudf
import cupy as cp
import pygcylon as gc

def local_merge():
    df1 = cudf.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})
    df2 = cudf.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})
    print("df1: \n", df1)
    print("df2: \n", df2)
    cdf1 = gc.DataFrame(df1)
    cdf2 = gc.DataFrame(df2)
    cdf3 = cdf1.merge(right=cdf2, how="left", on="first", left_index=False, right_index=False)
    print("locally merged df: \n", cdf3.df)


def dist_merge():
    env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
    print("CylonEnv Initialized: My rank: ", env.rank)

    df1 = cudf.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})
    df2 = cudf.DataFrame({'first': cp.random.randint(100, 110, 5), 'second': cp.random.randint(100, 110, 5)})

    arrays = [['a', 'a', 'b', 'b', 'c', 'c'], [1, 2, 3, 4, 5, 6]]
    tuples = list(zip(*arrays))
    idx = cudf.MultiIndex.from_tuples(tuples)
    # df1.index = idx
    # df2.index = idx
    print(df1)
    print(df2)
    cdf1 = gc.DataFrame(df1)
    cdf2 = gc.DataFrame(df2)
    cdf3 = cdf1.merge(right=cdf2, on="first", how="left", left_on=None, right_on=None, left_index=False, right_index=False, env=env)
    print("distributed joined df:\n", cdf3.df)
    env.finalize()

#####################################################
# local join test
local_merge()

# distributed join
# dist_merge()
