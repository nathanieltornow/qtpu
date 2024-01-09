partition(P) :- P = 0..2 - 1.

{ node_in_partition(Node, P) : partition(P) } == 1 :- node(Node, _).

partition_size(P, Size) :- partition(P), Size = #count{ Node : node_in_partition(Node, P) }.

partition_size_sum(Sum) :- Sum = #sum{ Size**2 : partition_size(_, Size) }.

cut_edge(Node1, Node2) :- 
    edge(Node1, Node2), 
    node_in_partition(Node1, P1), 
    node_in_partition(Node2, P2), 
    P1 != P2.

#minimize { Sum : partition_size_sum(Sum) }.

#show cut_edge/2.
