fragment(Frag) :- Frag = 0..num_fragments.

{ gate_in_frag(OpID, Frag) : fragment(Frag) } == 1 :- gate(OpID, _, _).
:- fragment(P), not gate_in_frag(_, P), P > 0.

cutted_wire(OpID1, OpID2, Qubit) :- 
    wire(OpID1, OpID2, Qubit), 
    gate_in_frag(OpID1, Frag1), 
    gate_in_frag(OpID2, Frag2), 
    Frag1 != Frag2, 
    Frag1 != 0, 
    Frag2 != 0.

num_wires_cut(NumWiresCut) :- NumWiresCut = #count { OpID1, Qubit : cutted_wire(OpID1, _, Qubit) }.

num_gates_cut(NumGatesCut) :- NumGatesCut = #count { OpID : gate_in_frag(OpID, 0) }.


wire(OpID1, OpID3, Qubit) :- 
    wire(OpID1, OpID2, Qubit), 
    wire(OpID2, OpID3, Qubit),
    gate_in_frag(OpID2, 0).




weighted_cuts(C) :- C = 4*W + 6*G, num_wires_cut(W), num_gates_cut(G).


gate_on_qubit(OpID, Qubit) :- gate(OpID, Qubit, _).
gate_on_qubit(OpID, Qubit) :- gate(OpID, _, Qubit).


qubit_in_fragment(Frag, Qubit) :- fragment(Frag), gate_on_qubit(G, Qubit), gate_in_frag(G, Frag), Frag != 0.





fragment_size(Frag, Size) :- fragment(Frag), Frag != 0, Size = #count{ Qubit : qubit_in_fragment(Frag, Qubit) }.

#minimize { Cuts : weighted_cuts(Cuts) }.

cutted_op(OpID) :- gate_in_frag(OpID, 0).

#show gate_in_frag/2.
