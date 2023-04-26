fragment(Frag) :- Frag = 0..num_fragments.

{ gate_in_frag(OpID, Frag) : fragment(Frag) } == 1 :- gate(OpID, _, _).
:- fragment(P), not gate_in_frag(_, P), P > 0.

cutted_wire(OpID1, OpID2, Qubit) :- wire(OpID1, OpID2, Qubit), gate_in_frag(OpID1, Frag1), gate_in_frag(OpID2, Frag2), Frag1 != Frag2, Frag1 != 0, Frag2 != 0.

num_wires_cut(NumWiresCut) :- NumWiresCut = #count { OpID1, Qubit : cutted_wire(OpID1, _, Qubit) }.
#minimize { NumWiresCut : num_wires_cut(NumWiresCut) }.

num_gates_cut(NumGatesCut) :- NumGatesCut = #count { OpID : op_in_frag(OpID, 0) }.
#minimize { NumGatesCut : num_gates_cut(NumGatesCut) }.

% num_qubits_in_frag(Frag, NumQubits) :- fragment(Frag), Frag != 0, NumQubits = #count { Qubit : op_in_frag(OpID, Frag), op_on_qubit(OpID, Qubit) }.
% num_qubits_diff(Frag1, Frag2, NumQubitsDiff) :- num_qubits_in_frag(Frag1, NumQubits1), num_qubits_in_frag(Frag2, NumQubits2), NumQubitsDiff = |NumQubits1 - NumQubits2|.
% num_qubits_diff_sum(NumQubitsDiffSum) :- NumQubitsDiffSum = #sum { NumQubitsDiff, Frag1, Frag2 : num_qubits_diff(Frag1, Frag2, NumQubitsDiff) }.
% #minimize { NumQubitsDiffSum : num_qubits_diff_sum(NumQubitsDiffSum) }.

cutted_op(OpID) :- op_in_frag(OpID, 0).

#show cutted_wire/3.
#show cutted_op/1.