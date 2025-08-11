from pysat.solvers import Solver
from itertools import combinations

# Global variable
top_id = 2

INPUT_FILE = 'input2.txt'  # Default input file path
CUR_WIDTH = 16

def solve_no_hole_anti_k_labeling(graph, k, width):
    # if width == 1:
    #     print("Width must be greater than 1")
    #     return
    global top_id
    solver = Solver(name='g4', with_proof=True)
    clauses = [[-1]]
    n = len(graph)
    block = (k - 1) // width
    last_block_size = k - (block) * width
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
            if block_id == 1:
                for index in range(1, width + 1):
                    tmp2.append(top_id)
                    top_id += 1
            else:
                for index in range(1, width):
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
                last_block_size = k - (block) * width
                for index in range(1, last_block_size + 1):
                    tmp2.append(top_id)
                    top_id += 1
            else:
                for index in range(1, width + 1):
                    tmp2.append(top_id)
                    top_id += 1
            tmp.append(tmp2)
        R.append(tmp)

    # clauses.extend(Symetry_breaking(graph, x))

    kk = len(clauses)
    # Encode SCL <= 1
    for i in range(1, n + 1):
    
        for block_id in range(1, block + 1):
            order = [-1]
            # order[index] replace x[i][label] for convenience in encoding since stair case is different between odd and even block_id
            if block_id == 1:
                for index in range(1, width + 1):
                    label = block_id * width - index + 1
                    order.append(x[i][label])
                clauses.extend(SCL_AMO(order, L, i, block_id, width))
            else:
                for index in range(1, width):
                    label = block_id * width - index + 1
                    order.append(x[i][label])
                clauses.extend(SCL_AMO(order, L, i, block_id, width - 1))
        
        for block_id in range(1, block + 1):
            
            if block_id == block:
                # Last block may not be full
                order = [-1]
                # order[index] replace x[i][label] for convenience in encoding since stair case is different between odd and even block_id
                for index in range(1, last_block_size + 1):
                    label = block_id * width + index
                    order.append(x[i][label])
                clauses.extend(SCL_AMO(order, R, i, block_id, last_block_size))
            else:    
                order = [-1]
                # order[index] replace x[i][label] for convenience in encoding since stair case is different between odd and even block_id
                for index in range(1, width + 1):
                    label = block_id * width + index
                    order.append(x[i][label])
                clauses.extend(SCL_AMO(order, R, i, block_id, width))
    print(len(clauses) - kk)
    # Exactly one
    kk = len(clauses)
    for i in range(1, n + 1):
        tmp = []
        # Collect variables for EO
        tmp.append(L[i][1][width])
        for block_id in range(1, block):
            tmp.append(R[i][block_id][width])
        tmp.append(R[i][block][last_block_size])
        clauses.extend(Exactly_One2(tmp))
    print(len(clauses) - kk)
    # Every label must be used at least once

    for label in range(1, k + 1):
        clause = []
        for i in range(1, n + 1):
            clause.append(x[i][label])
        clauses.append(clause)

    # Hai đỉnh được nối với nhau không được gán nhãn có hiệu tuyệt đối nhỏ hơn width
    # TODO: check this again
    # kk = len(clauses)

    for u in graph:
        for v in graph[u]:
            clauses.append([-L[u][1][width], -L[v][1][width]])
            for label in range(2, k - width + 2):
                block_id = (label - 1) // width + 1
                if label % width == 1:
                    clauses.append([-R[u][block_id - 1][width], -R[v][block_id - 1][width]])
                else:
                    lu = L[u][block_id][width - (label - 1) % width]
                    ru = R[u][block_id][(label - 1) % width]
                    
                    lv = L[v][block_id][width - (label - 1) % width]
                    rv = R[v][block_id][(label - 1) % width] 
                    
                    clauses.append([-lu, -lv])
                    clauses.append([-lu, -rv])
                    clauses.append([-ru, -lv])
                    clauses.append([-ru, -rv])
            
    # IMPORTANT: test code
    # for u in graph:
    #     for v in graph[u]:
    #         for labelu in range(1, k + 1):
    #             if labelu < width:
    #                 for labelv in range(1, labelu + 1):
    #                     clauses.append([-x[u][labelu], -x[v][labelv]])
    #             else:
    #                 for labelv in range(labelu - width + 1, labelu + 1):
    #                     clauses.append([-x[u][labelu], -x[v][labelv]])
    #                 for labelv in range(labelu + 1, min(k + 1, labelu + width)):
    #                     clauses.append([-x[u][labelu], -x[v][labelv]])

    # print(len(clauses) - kk)
    # Write clauses to a file
    with open("clauses_output.txt", "w") as file:
        for clause in clauses:
            file.write(" ".join(map(str, clause)) + " 0\n")

    # Print x, L, R at the beginning of the file
    with open("clauses_output.txt", "r+") as file:
        content = file.read()
        file.seek(0, 0)
        file.write("x:\n")
        for i in range(1, len(x)):
            file.write(" ".join(map(str, x[i])) + "\n")
        file.write("\nL:\n")
        for i in range(1, len(L)):
            for block_id in range(1, len(L[i])):
                file.write(" ".join(map(str, L[i][block_id])) + "\n")
        file.write("\nR:\n")
        for i in range(1, len(R)):
            for block_id in range(1, len(R[i])):
                file.write(" ".join(map(str, R[i][block_id])) + "\n")
        file.write(content)
    solver.append_formula(clauses)
    print(f"Number of variables: {solver.nof_vars()}")
    print(f"Number of clauses: {solver.nof_clauses()}")
    if solver.solve():
        for i in range(1, n + 1):
            for label in range(1, k + 1):
                if solver.get_model()[x[i][label] - 1] > 0:
                    print(f"Vertex {i} is assigned label {label}")
        print("Solution found:")
    else:
        print("No solution exists")

def SCL_AMO(order, R, i, block_id, width):
    # x <=> order
    # x1 <= 1 <=> R1
    # x1 + x2 <= 1 <=> R2
    # x1 + x2 + x3 <= 1 <=> R3
    # x1 + x2 + x3 + x4 <= 1 <=> R4

    clauses = []
    
    # Formula in 4.1
    # Formula (9)
    for index in range(1, width + 1):
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

def Symetry_breaking(graph, x):
    
    cnt = [0] * (len(graph) + 1)
    for u in graph:
        for v in graph[u]:
            cnt[u] += 1
            cnt[v] += 1
    node = -1
    for i in range(1, len(cnt)):
        if node == -1 or cnt[i] < cnt[node]:
            node = i
    
    clause = []
    for label in range(1, len(graph) // 2 + 1):
        clause.append([-x[node][label]])
    
    return clause

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

def Exactly_One2(variables):
    clauses = []
    # At least one
    clauses.append(variables)
    
    for i in variables:
        for j in variables:
            if i != j:
                clauses.append([-i, -j])
    
    return clauses

def read_input(file_path):
    graph = {}
    with open(file_path, 'r') as file:
        n, e = map(int, file.readline().split())
        for i in range(1, n + 1):
            graph[i] = []
        for _ in range(e):
            u, v = map(int, file.readline().split())
            graph[u].append(v)

    return graph

# Example usage
if __name__ == "__main__":
    # Example graph: Path with 4 vertices (0-1-2-3)
    graph = {
        1: [2, 3, 4],
        2: [3],
        3: [4],
        4: [],
        5: [6],
        6: [],
        7: [8],
        8: []
        # 10: [],
        # 11: [],
        # 12: [],
        # 13: [],
        # 14: []
    }
    graph = read_input(INPUT_FILE)
    k = len(graph)
    width = CUR_WIDTH   # Có thể thay đổi width nằm trong khoảng [1,k-1]
    solve_no_hole_anti_k_labeling(graph, k, width)
    # if solution:
    #     print(f"Found no-hole solution with minimum edge difference of {min_diff}:")
    #     used_labels = set(solution.values())
    #     print(f"Used labels: {sorted(used_labels)} (all labels from 1 to {k} are used)")
    #     for v in sorted(solution):
    #         print(f"Vertex {v}: Label {solution[v]}")
    # else:
    #     print("No solution exists")