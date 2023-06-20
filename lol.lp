gate(0).
gate_on_qubit(0, 0).
wire(0, 0, 8).
gate(8).
gate_on_qubit(8, 0).
gate_on_qubit(8, 1).
wire(0, 8, 15).
wire(1, 8, 15).
wire(1, 8, 9).
gate(1).
gate_on_qubit(1, 1).
wire(1, 1, 8).
gate(2).
gate_on_qubit(2, 2).
wire(2, 2, 9).
gate(9).
gate_on_qubit(9, 1).
gate_on_qubit(9, 2).
wire(1, 9, 15).
wire(2, 9, 10).
gate(3).
gate_on_qubit(3, 3).
wire(3, 3, 10).
gate(10).
gate_on_qubit(10, 2).
gate_on_qubit(10, 3).
wire(2, 10, 16).
wire(3, 10, 11).
gate(4).
gate_on_qubit(4, 4).
wire(4, 4, 11).
gate(11).
gate_on_qubit(11, 3).
gate_on_qubit(11, 4).
wire(3, 11, 17).
wire(4, 11, 12).
gate(5).
gate_on_qubit(5, 5).
wire(5, 5, 12).
gate(12).
gate_on_qubit(12, 4).
gate_on_qubit(12, 5).
wire(4, 12, 18).
wire(5, 12, 13).
gate(6).
gate_on_qubit(6, 6).
wire(6, 6, 13).
gate(13).
gate_on_qubit(13, 5).
gate_on_qubit(13, 6).
wire(5, 13, 19).
wire(6, 13, 14).
gate(7).
gate_on_qubit(7, 7).
wire(7, 7, 14).
gate(14).
gate_on_qubit(14, 6).
gate_on_qubit(14, 7).
wire(6, 14, 20).
wire(6, 14, 21).
wire(7, 14, 21).
gate(15).
gate_on_qubit(15, 0).
gate_on_qubit(15, 1).
wire(0, 15, 22).
wire(1, 15, 16).
gate(16).
gate_on_qubit(16, 1).
gate_on_qubit(16, 2).
wire(1, 16, 23).
wire(2, 16, 17).
gate(17).
gate_on_qubit(17, 2).
gate_on_qubit(17, 3).
wire(2, 17, 24).
wire(3, 17, 18).
gate(18).
gate_on_qubit(18, 3).
gate_on_qubit(18, 4).
wire(3, 18, 25).
wire(4, 18, 19).
gate(19).
gate_on_qubit(19, 4).
gate_on_qubit(19, 5).
wire(4, 19, 26).
wire(5, 19, 20).
gate(20).
gate_on_qubit(20, 5).
gate_on_qubit(20, 6).
wire(5, 20, 27).
wire(6, 20, 21).
gate(21).
gate_on_qubit(21, 6).
gate_on_qubit(21, 7).
wire(6, 21, 28).
wire(7, 21, 29).
gate(22).
gate_on_qubit(22, 0).
gate(23).
gate_on_qubit(23, 1).
gate(24).
gate_on_qubit(24, 2).
gate(25).
gate_on_qubit(25, 3).
gate(26).
gate_on_qubit(26, 4).
gate(27).
gate_on_qubit(27, 5).
gate(28).
gate_on_qubit(28, 6).
gate(29).
gate_on_qubit(29, 7).

    { vgate(Gate) : gate(Gate) }.
    
    :- N = #count{Gate : vgate(Gate)}, N > 3.
    
    path(Gate1, Gate2) :- wire(_, Gate1, Gate2), not vgate(Gate1).
    path(Gate1, Gate3) :- path(Gate1, Gate2), path(Gate2, Gate3).
    
    depends_on(Qubit1, Qubit2) :- 
        gate_on_qubit(Gate1, Qubit1),
        gate_on_qubit(Gate2, Qubit2),
        path(Gate1, Gate2),
        Qubit1 != Qubit2.
    
    num_deps(N) :- N = #count{Qubit1, Qubit2 : depends_on(Qubit1, Qubit2)}.
    
    num_vgates(N) :- N = #count{Gate : vgate(Gate)}.
    
    #minimize{N : num_deps(N)}.
    #minimize{N : num_vgates(N)}.
    #show vgate/1.