
pushfirst!(PyCall.PyVector(PyCall.pyimport("sys")."path"), srcdir())
psis = PyCall.pyimport("psis")

