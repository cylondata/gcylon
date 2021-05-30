import cupy as cp
import pygcylon as gc

env: gc.CylonEnv = gc.CylonEnv(config=gc.MPIConfig(), distributed=True)
print("CylonContext Initialized: My rank: ", env.rank)

start = 100 * env.rank
df = gc.DataFrame({'first': cp.random.randint(start, start + 10, 10),
                   'second': cp.random.randint(start, start + 10, 10)})
print("initial df from rank: ", env.rank, "\n", df)

hash_columns = [df._cdf._num_indices + 0]
shuffledDF = gc.frame.shuffle(df.to_cudf(), hash_columns, env)

print("shuffled df from rank: ", env.rank, "\n", shuffledDF)

env.finalize()
