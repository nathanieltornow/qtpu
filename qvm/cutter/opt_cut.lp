{ cut_edge(V1, V2) } :- edge(V1, V2).

:- N = #count { V1, V2 : cut_edge(V1, V2) }, N != 4.


% We don't want to cut edges that don't dissconnect the graph
:- cut_edge(V1, V2), connected(V1, V2).

connected(V1, V2) :- edge(V1, V2), not cut_edge(V1, V2), V1 != V2.
connected(V1, V3) :- connected(V1, V2), connected(V2, V3), V1 != V2.
connected(V1, V2) :- connected(V2, V1).
connected(V1, V1) :- vertex(V1, _).

% vertex_cost(V, 0) :- not cut_edge(V, _), not cut_edge(_, V).
cut_vertex(V) :- cut_edge(V, _).
cut_vertex(V) :- cut_edge(_, V).

vertex_cost(V1, Cost) :- vertex(V1, _), Cost = #count { V2 : cut_vertex(V2), connected(V1, V2) }.
max_cost(MaxCost) :- MaxCost = #max { Cost : vertex_cost(_, Cost) }.

% cut_vertex_cost(V1, 5) :- cut_edge(V1, _).
% cut_vertex_cost(V1, 5) :- cut_edge(_, V1).


% vertex_cost(V1, TotalCost) :- 
%     vertex(V1, _),
%     TotalCost = #sum { C, V2 : cut_vertex_cost(V2, C), connected(V1, V2) }.

% prevent the case that everything is cut
% :- vertex_cost(_, C), C = 0.    


num_connections(V1, N) :- vertex(V1, _), N = #count { V2 : connected(V1, V2) }.
num_conn_sum(Sum) :- Sum = #sum { N**2 : vertex(V1, _), num_connections(V1, N) }.


% :- num_conn_sum(Sum1), num_conn_sum(Sum2), Sum1 != Sum2.

#minimize { S@1 : num_conn_sum(S) }.
#minimize { C@1 : max_cost(C) }.
% #minimize { C@10000 : vertex_cost(_, C) }.

% max_cost(MaxCost) :- MaxCost = #max { TotalCost : vertex_cost(_, TotalCost) }.

% :- max_cost(MaxCost1), max_cost(MaxCost2), MaxCost1 != MaxCost2.
% #minimize { C@2 : max_cost(C) }.




% :- #count { V1, V2 : cut_edge(V1, V2) } != 1.



% #minimize { Sum : partition_size_sum(Sum) }.


% #show cut_edge/2.
