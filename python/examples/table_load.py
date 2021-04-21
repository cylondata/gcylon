import cudf
from pygcylon.ex import simple

inputFile = "data/cities.csv"
tbl = cudf.read_csv(inputFile)
tbl.info()

simple.rows(tbl)
