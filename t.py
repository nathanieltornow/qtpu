import multiprocessing

class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def evaluate_tree_parallel(root):
    if root is None:
        return 0
    
    # Evaluate the tree in parallel
    with multiprocessing.Pool() as pool:
        result = evaluate_node(root, pool)
    
    return result

def evaluate_node(node, pool):
    if node.left is None and node.right is None:
        return node.value
    
    if node.left is not None and node.right is not None:
        left_result = pool.apply_async(evaluate_node, (node.left, pool))
        right_result = pool.apply_async(evaluate_node, (node.right, pool))
        return perform_operation(node.value, left_result.get(), right_result.get())

def perform_operation(operator, left, right):
    if operator == '+':
        return left + right
    elif operator == '-':
        return left - right
    elif operator == '*':
        return left * right
    elif operator == '/':
        if right != 0:
            return left / right
        else:
            raise ValueError("Division by zero")

# Example usage
if __name__ == "__main__":
    # Example tree: (+ (+ 1 2) (* 3 4))
    leaf1 = Node(1)
    leaf2 = Node(2)
    leaf3 = Node(3)
    leaf4 = Node(4)
    inner_node1 = Node('+', leaf1, leaf2)
    inner_node2 = Node('*', leaf3, leaf4)
    root = Node('+', inner_node1, inner_node2)

    result = evaluate_tree_parallel(root)
    print("Result:", result)
