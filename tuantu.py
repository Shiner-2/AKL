from pysat.solvers import Solver
from itertools import combinations
from pysat.pb import PBEnc
import time
from pysat.card import CardEnc
import os
import glob
import multiprocessing
import pandas as pd
import sys
#  ./painless/build/release/painless_release cnf/anti_k_labeling_n39_k31_w15.cnf   -c=4   -solver=cckk -no-model

# Global variable
top_id = 2

def solve_no_hole_anti_k_labeling(graph, k, width, queue):
    log_file = open("log32.txt", "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    # if width <= 1:
    #     print("Width must be greater than 1!!!!!!!!!!!!")
    #     return False
    start = time.time()
    # if width == 1:
    #     print("Width must be greater than 1")
    #     return
    global top_id
    top_id = 2
    solver = Solver(name='Cadical195')
    clauses = [[-1]]
    n = len(graph)
    # Each block container width number of labels
    # Create a 2D list to hold the labels for each vertex
    x = [[1]]
    # x[i][label] i là số thứ tự của đỉnh, label là nhãn được gán của đỉnh nó x[i][label] = 1 nghĩa là nhãn label được gán cho đỉnh i
    for i in range(1, n + 1):
        tmp = [1]
        for j in range(1, k + 1):
            tmp.append(top_id)
            top_id += 1
        x.append(tmp)
    
    L = [[1]]
    # L[i][j] từ trái sang phải, nếu có ít nhất 1 nhãn trong block j được gán cho đỉnh i thì L[i][j] = 1
    for i in range(1, n + 1):
        tmp = [1]
        for j in range(1, k + 1):
            tmp.append(top_id)
            top_id += 1
        L.append(tmp)

    R = [[1]]
    # R[i][j] từ phải sang trái, nếu có ít nhất 1 nhãn trong block j được gán cho đỉnh i thì R[i][j] = 1
    for i in range(1, n + 1):
        tmp = [1]
        for j in range(1, k + 1):
            tmp.append(top_id)
            top_id += 1
        R.append(tmp)

    clauses.extend(Symetry_breaking(graph, x))

    # Link x and L

    for i in range(1, n + 1):
        xx = [1]
        LL = [1]
        for label in range(1, k + 1):
            xx.append(x[i][label])
            LL.append(L[i][label])
        clauses.extend(SCL_AMO(xx, LL, k))

    # Link x and R
    for i in range(1, n + 1):
        xx = [1]
        RR = [1]
        for label in range(1, k + 1):
            RR.append(R[i][label])
        for label in range(k, 0, -1):
            xx.append(x[i][label])
        clauses.extend(SCL_AMO(xx, RR, k))

    # Link L and R
    # for i in range(1, n + 1):
    #     for label in range(1, k + 1):
    #         clauses.append([-L[i][label], R[i][k - label + 1]])
    #         clauses.append([-R[i][k - label + 1], L[i][label]])

    # print(len(clauses) - kk)
    # Exactly one
    kk = len(clauses)
    for i in range(1, n + 1):
        tmp = []
        # Collect variables for EO
        for label in range(1, k + 1):
            tmp.append(x[i][label])
        clauses.extend(Exactly_One1(tmp))
    # print(len(clauses) - kk)

    # Every label must be used at least once

    for label in range(1, k + 1):
        clause = []
        for i in range(1, n + 1):
            clause.append(x[i][label])

        clauses.append(clause)

    # Label cant be use more than n-k+1 times
    # for label in range(1, k + 1):
    #     variables = []
    #     for i in range(1, n + 1):
    #         variables.append(x[i][label])
    #     clauses.extend(At_Most_K(variables, 4))

    # Hai đỉnh được nối với nhau không được gán nhãn có hiệu tuyệt đối nhỏ hơn width
    # TODO: check this again
    # kk = len(clauses)



    for u in graph:
        for v in graph[u]:
            for labelu in range(1, k + 1):
                minv = max(0, labelu - width)
                maxv = min(k + 1, labelu + width)
                if minv == 0 and maxv == k + 1:
                    continue
                if minv == 0:
                    clauses.append([-x[u][labelu], R[v][k - labelu - width + 1]])
                    clauses.append([-x[u][labelu], -L[v][labelu + width - 1]])
                if maxv == k + 1:
                    clauses.append([-x[u][labelu], L[v][labelu - width]])
                    clauses.append([-x[u][labelu], -R[v][k - labelu + width]])
                if minv > 0 and maxv < k + 1:
                    clauses.append([-x[u][labelu], L[v][labelu - width], R[v][k - labelu - width + 1]])


            for labelv in range(1, k + 1):
                minu = max(0, labelv - width)
                maxu = min(k + 1, labelv + width)
                if minv == 0 and maxv == k + 1:
                    continue
                if minu == 0:
                    clauses.append([-x[v][labelv], R[u][k - labelv - width + 1]])
                    clauses.append([-x[v][labelv], -L[u][labelv + width - 1]])
                if maxu == k + 1:
                    clauses.append([-x[v][labelv], L[u][labelv - width]])
                    clauses.append([-x[v][labelv], -R[u][k - labelv + width]])
                if minu > 0 and maxu < k + 1:
                    clauses.append([-x[v][labelv], L[u][labelv - width], R[u][k - labelv - width + 1]])



    # IMPORTANT: test code
    # for u in graph:
    #     for v in graph[u]:
    #         for labelu in range(1, k - width + 2):
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
    # with open("clauses_output.txt", "w") as file:
    #     for clause in clauses:
    #         file.write(" ".join(map(str, clause)) + " 0\n")

    # # Print x, L, R at the beginning of the file
    # with open("clauses_output.txt", "r+") as file:
    #     content = file.read()
    #     file.seek(0, 0)
    #     file.write("x:\n")
    #     for i in range(1, len(x)):
    #         file.write(" ".join(map(str, x[i])) + "\n")
    #     file.write("\nL:\n")
    #     for i in range(1, len(L)):
    #         for block_id in range(1, len(L[i])):
    #             file.write(" ".join(map(str, L[i][block_id])) + "\n")
    #     file.write("\nR:\n")
    #     for i in range(1, len(R)):
    #         for block_id in range(1, len(R[i])):
    #             file.write(" ".join(map(str, R[i][block_id])) + "\n")
    #     file.write(content)

    
    solver.append_formula(clauses)
    print(f"Number of variables: {solver.nof_vars()}", flush=True)
    print(f"Number of clauses: {solver.nof_clauses()}", flush=True)
    queue.put(solver.nof_vars())
    queue.put(solver.nof_clauses())
    # Write DIMACS CNF instead of solving
    folder_path = "C:/Users/Admin/Desktop/Lab/AKL/cnf"
    folder_path = os.path.join(folder_path, f"K_n{n}_k{k}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    cnf_filename = os.path.join(folder_path, f"K_n{n}_k{k}_w{width}.cnf")
    with open(cnf_filename, "w") as f:
        f.write(f"p cnf {solver.nof_vars()} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")
    print(f"CNF written to {cnf_filename}")
    return

    # if width == 15:
    #     write_cnf_file(clauses, solver, n, k, width)

    if solver.solve():
        queue.put(True)
        # for i in range(1, n + 1):
        #     for label in range(1, k + 1):
        #         if solver.get_model()[x[i][label] - 1] > 0:
        #             print(f"Vertex {i} is assigned label {label}")
        print(f"Solution found: {width}", flush=True)
        end = time.time()
        print(f"Time taken: {end - start} seconds", flush=True)
        return True
    else:
        queue.put(False)
        print("No solution exists", flush=True)
        end = time.time()
        print(f"Time taken: {end - start} seconds", flush=True)
        return False

def SCL_AMO(x, R, k):
    # x <=> order
    # x1 <= 1 <=> R1
    # x1 + x2 <= 1 <=> R2
    # x1 + x2 + x3 <= 1 <=> R3
    # x1 + x2 + x3 + x4 <= 1 <=> R4

    clauses = []
    
    # Formula in 4.1
    # Formula (9)
    for index in range(1, k + 1):
        clauses.append([-x[index], R[index]])

    # Formula (10)
    for index in range(1, k):
        clauses.append([-R[index], R[index + 1]])

    # Formula (11)
    clauses.append([x[1], -R[1]])

    # Formula (12)
    for index in range(2, k + 1):
        clauses.append([x[index], R[index-1], -R[index]])

    # Formula (13)
    for index in range(2, k + 1):
        clauses.append([-x[index], -R[index-1]])

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

def Exactly_One1(variables):
        # Use BDD-based encoding for Exactly One constraint
    lits = variables
    weights = [1] * len(variables)
    clauses = []
    bound = 1
    global top_id
    cnf1 = PBEnc.atleast(lits=lits, weights=weights, bound=bound, encoding=1, top_id=top_id)
    top_id = cnf1.nv
    for clause in cnf1.clauses:
        clauses.append(clause)

    cnf2 = PBEnc.atmost(lits=lits, weights=weights, bound=bound, encoding=1, top_id=top_id)
    top_id = cnf2.nv
    for clause in cnf2.clauses:
        clauses.append(clause)

    return clauses

def Exactly_One3(variables, encoding=1):
    # Exactly one variable in 'variables' must be True using cardinality encoding
    global top_id
    cnf = CardEnc.equals(lits=variables, bound=1, encoding=encoding, top_id=top_id)
    top_id = cnf.nv
    return list(cnf.clauses)

def At_Most_K(variables, k,encoding=1):
    # At most k variables in 'variables' must be True using cardinality encoding
    global top_id
    cnf = CardEnc.atmost(lits=variables, bound=k, encoding=encoding, top_id=top_id)
    top_id = cnf.nv
    return list(cnf.clauses)

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

def run_test_with_timeout(graph, k, width, timeout_sec=1800):
    log_file = open("log32.txt", "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    global res2
    start = time.time()
    queue = multiprocessing.Queue()

    p = multiprocessing.Process(target=solve_no_hole_anti_k_labeling, args=(graph, k, width, queue))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        #print(f"[Test {filename, c}] Quá thời gian {timeout_sec} giây!")
        p.terminate()
        p.join()
    num_var = queue.get() if not queue.empty() else None
    num_clause = queue.get() if not queue.empty() else None
    verdict = queue.get() if not queue.empty() else None
    res2.extend([num_var, num_clause, verdict])
    end = time.time()
    elapsed = round((end-start), 2)
    print(f"[Test k={k}, w={width}] Time: {elapsed} seconds", flush=True)
    if verdict == True:
        print(f"Found a solution {width}", flush=True)
        return True
    else:
        print(f"No solution found {width}", flush=True)
        return False

res = [["filename", "k", "width", "num_vars", "num_clauses", "verdict", "time"]]
res2 = []

def binary_search_for_ans(graph, k, left, right, file, timeout_sec=1800):
    log_file = open("log32.txt", "a", encoding="utf-8", buffering=1)
    global res
    global res2
    res.append([file, None, None, None, None, None, None])
    time_left = timeout_sec
    ans = -9999
    while left <= right:
        width = left
        time_start = time.time()
        res2.extend([file, k, width])
        if run_test_with_timeout(graph, k, width, time_left):
            res2.append(round(time.time() - time_start, 2))
            res.append(res2)
            res2 = []
            time_left -= time.time() - time_start
            ans = width
            left = width + 1
            # print("$$$$")
            # print("SAT")
            # print("$$$$")
            # # Delete later
            # break
            # # Delete later
            if time_left <= 0.5:
                if ans == -9999:
                    return -9999
                return -ans
        else:
            # print("$$$$")
            # print("UNSAT")
            # print("$$$$")
            
            # # Delete later
            # break
            # # Delete later

            res2.append(round(time.time() - time_start, 2))
            res.append(res2)
            res2 = []
            time_left -= time.time() - time_start
            if time_left <= 0.5:
                if ans == -9999:
                    return -9999
                return -ans
            break
            
    return ans

def write_to_excel(data, output_file='output/output33.xlsx'):
    log_file = open("log32.txt", "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    try:
        df = pd.DataFrame(data)
        df.to_excel(output_file, index=False)
        print(f"Data written to {output_file}")
    except Exception as e:
        print(f"Error writing to Excel: {e}")

def cnf():
    log_file = open("log32.txt", "w", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    folder_path = "data/11. hb"
    files = glob.glob(f"{folder_path}/*")
    lst = []
    filename = []
    upper_bound = [220,220,136,24,17,22,39,9,35,79,112,13,51,64,104,9,8,102,36,326,7,256,14]
    lower_bound = [168,172,115,19,11,15,25,3,2, 51, 83, 5,43, 2,  2,2,2, 94,22,  2,2,  2, 2]
    for file in files:
        lst.append(folder_path + "/" + os.path.basename(file))
        filename.append(os.path.basename(file))

    for i in range(10,len(lst)):
        time_start = time.time()
        graph = read_input(lst[i])
        k = len(graph)-upper_bound[i]//2
        left = lower_bound[i]
        left = 38
        right = k - 1
        file = filename[i]
        ans = -9999
        time_limit = 30
        ans = binary_search_for_ans(graph, k, left, right, file, time_limit)
        print("$$$$")
        print(ans)
        print("$$$$")
        time_end = time.time()
        print(f"Time taken for {file} with k = {k}: {time_end - time_start} seconds")
        if ans >= 0:
            print(f"Maximum width for {file} is {ans}")
        else:
            if ans == -9999:
                print(f"No answer before timeout for {file}")
            else:
                print(f"Maximum width before timeout for {file} is {-ans}")
            print("time out")
    write_to_excel(res)

def write_cnf_file(clauses, solver, n, k, width, filename="output.cnf"):
    """
    Write clauses to a CNF file in DIMACS format
    Args:
        clauses: List of clauses
        solver: SAT solver instance containing variable info
        n: Number of vertices
        k: Number of labels 
        width: Width parameter
        filename: Output filename
    """
    folder_path = "cnf"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    output_path = os.path.join(folder_path, f"anti_k_labeling_n{n}_k{k}_w{width}.cnf")
    
    with open(output_path, "w") as f:
        # Write header
        f.write(f"p cnf {solver.nof_vars()} {len(clauses)}\n")
        
        # Write clauses
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")
            
    print(f"CNF written to {output_path}")


# Example usage
if __name__ == "__main__":

    cnf()
    # graph = read_input(INPUT_FILE)  # Replace with your input file path
    # k = len(graph)
    # for width in CUR_WIDTH:     # Có thể thay đổi width nằm trong khoảng [1,k-1]
    #     solve_no_hole_anti_k_labeling(graph, k, width)
    # if solution:
    #     print(f"Found no-hole solution with minimum edge difference of {min_diff}:")
    #     used_labels = set(solution.values())
    #     print(f"Used labels: {sorted(used_labels)} (all labels from 1 to {k} are used)")
    #     for v in sorted(solution):
    #         print(f"Vertex {v}: Label {solution[v]}")
    # else:
    #     print("No solution exists")