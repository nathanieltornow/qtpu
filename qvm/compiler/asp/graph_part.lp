partition(P) :- P = 0..num_partitions - 1.

{ node_in_partition(Node, P) : partition(P) } == 1 :- node(Node).

partition_size(P, Size) :- 
    partition(P),
    Size = #count{ Node : node_in_partition(Node, P) }.

partition_size_sum(Sum) :- 
    Sum = #sum{ Size**2 : partition_size(_, Size) }.


