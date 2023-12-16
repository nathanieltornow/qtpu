from pytket.passes import DecomposeMultiQubitsCX
from pytket.circuit import Circuit


pass1 = DecomposeMultiQubitsCX()

circ = Circuit(3)
circ.CRz(0.5, 0, 1)
circ.T(2)
circ.add_barrier([0, 1, 2], data="v_cx")
circ.CSWAP(2, 0, 1)



for com in circ.get_commands():
    if com.op.get_name() == "Barrier":
        print(com.op.data)
    print(com.op)

# from pytket.predicates import CompilationUnit
# from pytket.predicates import GateSetPredicate
# from pytket.circuit import OpType

# pred1 = GateSetPredicate({OpType.Rz, OpType.X, OpType.CX})
# cu = CompilationUnit(circ, [pred1])
# pass1.apply(cu)
# pass1.apply(cu)
# circ1 = cu.circuit


# print(circ1.get_commands())
# print(cu.check_all_predicates())
# pass1.apply(cu)
# print(cu.check_all_predicates())