partition(P) :- P = 0..num_partitions-1.

% prevent one edge from having multiple weights
:- edge(X, Y, W1), edge(X, Y, W2), W1 != W2.
% make the graph undirected
edge(X, Y, W) :- edge(Y, X, W).

% every assignment from a vertex to a partition
{ partition(VertexID, P) : partition(P) } == 1 :- vertex(VertexID, _).
% make sure that each partition has at least one vertex
:- partition(P), not partition(_, P).

% the total weight of a partition
partition_weight(P, WeightSum) :- partition(P), WeightSum = #sum { Weight, VertexID : vertex(VertexID, Weight), partition(VertexID, P) }.

% all the pairs of partitions
partition_pair(P1, P2) :- partition(P1), partition(P2), P1 < P2.

% the difference in weight between two partitions
% vertex_weight_difference(P1, P2, Diff) :- partition_pair(P1, P2), Diff = |W1 - W2|, partition_weight(P1, W1), partition_weight(P2, W2).
% the sum of the differences in weight between all pairs of partitions
% vertex_weight_diff_sum(DiffSum) :- 
    % DiffSum = #sum { Diff, P1, P2 : vertex_weight_difference(P1, P2, Diff) }.

% % the weight of an edge between two partitions
edge_weight(P1, P2, WeightSum) :- 
    partition_pair(P1, P2), 
    WeightSum= #sum { Weight, VertexID1, VertexID2 : edge(VertexID1, VertexID2, Weight), partition(VertexID1, P1), partition(VertexID2, P2) }.
% aggregated sum of the weight of all edges between two partitions
num_cuts(WeightSum) :-
    WeightSum = #sum { W, VertexID1, VertexID2 : edge_weight(VertexID1, VertexID2, W) }.

num_vertices(P, N) :- partition(P), N = #count { VertexID : partition(VertexID, P), vertex(VertexID, _) }.



% optimization
% minimize the sum of the differences in weight between all pairs of partitions
% #minimize { DiffSum : vertex_weight_diff_sum(DiffSum) }.
% minimize the sum of the weight of all edges between all partitions pairs
#minimize { Cuts : num_cuts(Cuts) }.

#show num_cuts/1.
#show partition/2.