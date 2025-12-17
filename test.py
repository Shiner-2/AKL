from pysat.solvers import Solver
from itertools import combinations
from pysat.card import CardEnc

def AtLeastOne(variables):

    enc = CardEnc.atleast(
        lits=variables,
        bound=1,
        encoding=1
    )
    top_id = enc.nv

    return enc.clauses

solver = Solver(name='g3')
clauses = []
clauses.extend(AtLeastOne([1, 2, 3, 4, 5, 6]))
solver.append_formula(clauses)
print("Number of clauses after AtLeastOne:", solver.nof_clauses())