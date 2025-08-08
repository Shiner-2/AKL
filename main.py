from pysat.solvers import Solver
from itertools import combinations

# Global variable
top_id = 2


def solve_no_hole_anti_k_labeling(graph, k, width):
    # if width == 1:
    #     print("Width must be greater than 1")
    #     return
    global top_id
    solver = Solver(name='g3')
    clauses = [[-1]]
    n = len(graph)
    block = (k - 1) // width
    # Each block container width number of labels
    # Create a 2D list to hold the labels for each vertex
    x = [[1]]
    # x[i][label] i là số thứ tự của đỉnh, label là nhãn được gán của đỉnh nó x[i][label] = 1 nghĩa là nhãn label được gán cho đỉnh i
    for i in range(1, n + 1):
        tmp = [1]
        for label in range(1, k + 1):
            tmp.append(top_id)
            top_id += 1
        x.append(tmp)
    
    L = [[[1]]]
    # L[i][block_id][index] left block, i là số thứ tự của đỉnh, block_id là id của block, index là vị trí trong block
    for i in range(1, n + 1):
        tmp = [[1]]
        for block_id in range(1, block + 1):
            tmp2 = [1]
            if block_id == block:
                # Last block may not be full
                last_block_size = k - (block - 1) * width
                for index in range(1, last_block_size + 1):
                    tmp2.append(top_id)
                    top_id += 1
            else:
                for index in range(1, width + 1):
                    tmp2.append(top_id)
                    top_id += 1
            tmp.append(tmp2)
        L.append(tmp)

    R = [[[1]]]
    # R[i][block_id][index] right block i là số thứ tự của đỉnh, block_id là id của block, index là vị trí trong block
    for i in range(1, n + 1):
        tmp = [[1]]
        for block_id in range(1, block + 1):
            tmp2 = [1]
            if block_id == block:
                # Last block may not be full
                last_block_size = k - (block - 1) * width
                for index in range(1, last_block_size + 1):
                    tmp2.append(top_id)
                    top_id += 1
            else:
                for index in range(1, width + 1):
                    tmp2.append(top_id)
                    top_id += 1
            tmp.append(tmp2)
        R.append(tmp)

    # Encode SCL <= 1
    for i in range(1, n + 1):
        # TODO: block cuối có thể không đầy đủ fix sau
        for block_id in range(1, block + 1):
            order = [-1]
            # order[index] replace x[i][label] for convenience in encoding since stair case is different between odd and even block_id
            for index in range(1, width + 1):
                label = block_id * width - index + 1
                order.append(x[i][label])
        
            clauses.extend(SCL_AMO(order, L, i, block_id, width))
        
        for block_id in range(1, block + 1):
            order = [-1]
            # order[index] replace x[i][label] for convenience in encoding since stair case is different between odd and even block_id
            for index in range(1, width + 1):
                label = block_id * width + index
                order.append(x[i][label])
        
            clauses.extend(SCL_AMO(order, R, i, block_id, width))

    # TODO: check C again 
    C = [[1]]
    for i in range(1, n + 1):
        tmp = [1]
        for label in range(1, k + 1):
            tmp.append(top_id)
            top_id += 1
        C.append(tmp)


    for i in range(1, n + 1):
        for block_id in range(1, block + 1):
            # TODO: this part is sus
            for index in range(0, width + 1):
                clauses.append([-C[i][(block_id - 1) * width + index + 1], L[i][block_id][width - index], R[i][block_id][index]])
                clauses.append([C[i][(block_id - 1) * width + index + 1], -L[i][block_id][width - index]])
                clauses.append([C[i][(block_id - 1) * width + index + 1], -R[i][block_id][index]])

    # Exactly one
    for i in range(1, n + 1):
        tmp = []
        # Collect variables for EO
        for label in range(1, k + 1):
            tmp.append(x[i][label])
        clauses.extend(Exactly_One(tmp))

    # Every label must be used at least once

    for label in range(1, k + 1):
        clause = []
        for i in range(1, n + 1):
            clause.append(x[i][label])
        clauses.append(clause)

    # Hai đỉnh được nối với nhau không được gán nhãn có hiệu tuyệt đối nhỏ hơn width

    for u in graph:
        for v in graph[u]:
            for label in range(1, k - width + 2):
                clauses.append([-C[u][label], -C[v][label]])
                

    # print(clauses)
    # TODO: Code lại sau hiểu nhầm staircase

    solver.append_formula(clauses)
    if solver.solve():
        for i in range(1, n + 1):
            for label in range(1, k + 1):
                if solver.get_model()[x[i][label] - 1] > 0:
                    print(f"Vertex {i} is assigned label {label}")
    else:
        print("No solution exists")

def SCL_AMO(order, R, i, block_id, width):
    clauses = []
    
    # Formula in 4.1
    # Formula (9)
    for index in range(1, width + 1):
        label = block_id * width - index + 1
        clauses.append([-order[index], R[i][block_id][index]])

    # Formula (10)
    for index in range(1, width):
        clauses.append([-R[i][block_id][index], R[i][block_id][index + 1]])

    # Formula (11)
    clauses.append([order[1], -R[i][block_id][1]])

    # Formula (12)
    for index in range(2, width + 1):
        clauses.append([order[index], R[i][block_id][index-1], -R[i][block_id][index]])

    # Formula (13)
    for index in range(2, width + 1):
        clauses.append([-order[index], -R[i][block_id][index-1]])

    return clauses

def Exactly_One(variables):
    clauses = []
    global top_id

    # At least one
    clauses.append(variables)

    # At Most One
    R = []
    for i in range(len(variables)):
        R.append(top_id)
        top_id += 1
    
    for i in range(len(variables)):
        clauses.append([-variables[i], R[i]])

    for i in range(1, len(variables)):
        clauses.append([-variables[i], -R[i-1]])

    clauses.append([variables[0], -R[0]])

    for i in range(1, len(variables)):
        clauses.append([variables[i], R[i-1], -R[i]])

    for i in range(len(variables)-1):
        clauses.append([-R[i], R[i+1]])

    return clauses

# Example usage
if __name__ == "__main__":
    # Example graph: Path with 4 vertices (0-1-2-3)
    graph = {
        1: [2,3,4,5,6,8],
        2: [3],
        3: [4],
        4: [5],
        5: [6],
        6: [7],
        7: [8],
        8: [9],
        9: []
        # 10: [],
        # 11: [],
        # 12: [],
        # 13: [],
        # 14: []
    }
    k = 9
    width = 3     # Có thể thay đổi width nằm trong khoảng [1,k-1]
    solve_no_hole_anti_k_labeling(graph, k, width)
    # if solution:
    #     print(f"Found no-hole solution with minimum edge difference of {min_diff}:")
    #     used_labels = set(solution.values())
    #     print(f"Used labels: {sorted(used_labels)} (all labels from 1 to {k} are used)")
    #     for v in sorted(solution):
    #         print(f"Vertex {v}: Label {solution[v]}")
    # else:
    #     print("No solution exists")