# -*- coding: utf-8 -*-
from docplex.mp.model import Model
import time, os, glob, multiprocessing, pandas as pd, sys

# =========================
# L/R + SCL_AMO (MILP CPLEX)
# =========================

def solve_no_hole_anti_k_labeling_cplex(graph, k, width, queue, timelimit=None, threads=None):
    """
    Bản CPLEX bám sát SAT:
      - x[i,l] nhị phân như cũ
      - L[i][block_id][idx], R[i][block_id][idx] giống cách tạo ở SAT
      - SCL_AMO(order, T, i, block_id, w) giữ nguyên cấu trúc (9)-(13) nhưng tuyến tính hoá:
          (9)   (-order[idx] v  T[idx])       -> T[idx] >= order[idx]
          (10)  (-T[j] v T[j+1])              -> T[j+1] >= T[j]
          (11)  ( order[1] v -T[1])           -> T[1] <= order[1]
          (12)  ( order[j] v T[j-1] v -T[j])  -> T[j] <= order[j] + T[j-1]
          (13)  (-order[j] v -T[j-1])         -> T[j-1] <= 1 - order[j]
      - ExactlyOne trên các “điểm cuối” giống hệt SAT: sum(endpoints) == 1
      - Every label used ≥ 1: sum_i x[i,l] ≥ 1
      - Anti-width qua L/R như SAT: mỗi mệnh đề (¬a ∨ ¬b) -> a + b ≤ 1
    """
    log_file = open("log_random.txt", "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file

    n = len(graph)
    if width <= 1:
        print("Width must be greater than 1!!!!!!!!!!!!", flush=True)
        if queue is not None:
            queue.put(0); queue.put(0); queue.put(False)
        return False

    start = time.time()
    m = Model(name=f"NH-AKL_k{k}_w{width}", log_output=True)

    if timelimit is not None:
        m.time_limit = timelimit
    if threads is not None:
        try:
            m.context.cplex_parameters.threads = threads
        except Exception:
            pass

    V = range(1, n + 1)
    LBL = range(1, k + 1)

    # --- Biến x[i,l] ---
    x = {(i, l): m.binary_var(name=f"x_{i}_{l}") for i in V for l in LBL}

    # --- Chia block như SAT ---
    block = (k - 1) // width
    last_block_size = k - block * width  # (có thể < width)
    # Tránh corner khi k = width (block=??): theo code SAT, block >= 1 khi k>=2
    block = max(1, block)
    last_block_size = max(1, last_block_size)

    # --- Tạo L[i][b][idx] và R[i][b][idx] giống kích thước SAT ---
    # L: block 1 có 'width' biến; các block >1 có 'width-1'
    L_aux = {}  # (i,b,idx)
    for i in V:
        for b in range(1, block + 1):
            if b == 1:
                wlen = width
            else:
                wlen = max(1, width - 1)
            for idx in range(1, wlen + 1):
                L_aux[(i, b, idx)] = m.binary_var(name=f"L_{i}_{b}_{idx}")

    # R: block cuối có 'last_block_size'; các block khác có 'width'
    R_aux = {}
    for i in V:
        for b in range(1, block + 1):
            rlen = last_block_size if b == block else width
            for idx in range(1, rlen + 1):
                R_aux[(i, b, idx)] = m.binary_var(name=f"R_{i}_{b}_{idx}")

    # --- Hàm thêm ràng buộc SCL_AMO (công thức (9)-(13) tuyến tính hoá) ---
    def add_SCL_AMO(order_vars, T_tensor, i, b, wlen, side='R'):
        # (9): T[idx] >= order[idx]
        for idx in range(1, wlen + 1):
            if side == 'L':
                m.add_constraint(L_aux[(i, b, idx)] >= order_vars[idx], ctname=f"S9_L_{i}_{b}_{idx}")
            else:
                m.add_constraint(R_aux[(i, b, idx)] >= order_vars[idx], ctname=f"S9_R_{i}_{b}_{idx}")

        # (10): T[idx+1] >= T[idx]
        for idx in range(1, wlen):
            if side == 'L':
                m.add_constraint(L_aux[(i, b, idx + 1)] >= L_aux[(i, b, idx)], ctname=f"S10_L_{i}_{b}_{idx}")
            else:
                m.add_constraint(R_aux[(i, b, idx + 1)] >= R_aux[(i, b, idx)], ctname=f"S10_R_{i}_{b}_{idx}")

        # (11): T[1] <= order[1]
        if side == 'L':
            m.add_constraint(L_aux[(i, b, 1)] <= order_vars[1], ctname=f"S11_L_{i}_{b}")
        else:
            m.add_constraint(R_aux[(i, b, 1)] <= order_vars[1], ctname=f"S11_R_{i}_{b}")

        # (12): T[j] <= order[j] + T[j-1], j>=2
        for j in range(2, wlen + 1):
            if side == 'L':
                m.add_constraint(L_aux[(i, b, j)] <= order_vars[j] + L_aux[(i, b, j - 1)], ctname=f"S12_L_{i}_{b}_{j}")
            else:
                m.add_constraint(R_aux[(i, b, j)] <= order_vars[j] + R_aux[(i, b, j - 1)], ctname=f"S12_R_{i}_{b}_{j}")

        # (13): T[j-1] <= 1 - order[j], j>=2
        for j in range(2, wlen + 1):
            if side == 'L':
                m.add_constraint(L_aux[(i, b, j - 1)] <= 1 - order_vars[j], ctname=f"S13_L_{i}_{b}_{j}")
            else:
                m.add_constraint(R_aux[(i, b, j - 1)] <= 1 - order_vars[j], ctname=f"S13_R_{i}_{b}_{j}")

    # --- SCL_AMO cho từng đỉnh và từng block: y hệt cách tạo 'order' trong SAT ---
    for i in V:
        # Left side (L)
        for b in range(1, block + 1):
            order = {0: None}
            if b == 1:
                # width phần tử: label = b*width - idx + 1
                for idx in range(1, width + 1):
                    label = b * width - idx + 1
                    if 1 <= label <= k:
                        order[idx] = x[(i, label)]
                    else:
                        # nếu label vượt biên (hiếm), dùng 0 để không ảnh hưởng
                        order[idx] = m.binary_var(lb=0, ub=0, name=f"dummyL_{i}_{b}_{idx}")
                add_SCL_AMO(order, L_aux, i, b, width, side='L')
            else:
                # width-1 phần tử
                wlen = max(1, width - 1)
                for idx in range(1, wlen + 1):
                    label = b * width - idx + 1
                    if 1 <= label <= k:
                        order[idx] = x[(i, label)]
                    else:
                        order[idx] = m.binary_var(lb=0, ub=0, name=f"dummyL_{i}_{b}_{idx}")
                add_SCL_AMO(order, L_aux, i, b, wlen, side='L')

        # Right side (R)
        for b in range(1, block + 1):
            order = {0: None}
            if b == block:
                # Last block: last_block_size phần tử; label = b*width + idx
                rlen = last_block_size
                for idx in range(1, rlen + 1):
                    label = b * width + idx
                    if 1 <= label <= k:
                        order[idx] = x[(i, label)]
                    else:
                        order[idx] = m.binary_var(lb=0, ub=0, name=f"dummyR_{i}_{b}_{idx}")
                add_SCL_AMO(order, R_aux, i, b, rlen, side='R')
            else:
                # Các block khác: width phần tử; label = b*width + idx
                for idx in range(1, width + 1):
                    label = b * width + idx
                    if 1 <= label <= k:
                        order[idx] = x[(i, label)]
                    else:
                        order[idx] = m.binary_var(lb=0, ub=0, name=f"dummyR_{i}_{b}_{idx}")
                add_SCL_AMO(order, R_aux, i, b, width, side='R')

    # --- ExactlyOne trên các “điểm cuối” (giống SAT) ---
    for i in V:
        endpoints = []
        # L[i][1][width]
        endpoints.append(L_aux[(i, 1, width)])
        # R[i][b][width] cho b=1..block-1 (nếu width hợp lệ ở b đó)
        for b in range(1, block):
            if (i, b, width) in R_aux:
                endpoints.append(R_aux[(i, b, width)])
        # R[i][block][last_block_size]
        endpoints.append(R_aux[(i, block, last_block_size)])
        m.add_constraint(m.sum(endpoints) == 1, ctname=f"EO_endpoints_{i}")

    # --- Every label used ≥ 1 (No-hole) ---
    for l in LBL:
        m.add_constraint(m.sum(x[(i, l)] for i in V) >= 1, ctname=f"nohole_{l}")

    # --- Symmetry breaking (giữ đúng tinh thần SAT: x[node][l]=0 cho l<=n//2) ---
    deg = {i: 0 for i in V}
    for u in graph:
        for v in graph[u]:
            deg[u] += 1
            deg[v] += 1
    node = min(V, key=lambda t: deg[t])
    for l in range(1, n // 2 + 1):
        if (node, l) in x:
            m.add_constraint(x[(node, l)] == 0, ctname=f"sb_{node}_{l}")

    # --- Anti-width bằng L/R như SAT ---
    # Dịch từng mệnh đề (¬a ∨ ¬b) -> a + b ≤ 1
    # 1) mệnh đề đầu: [-L[u][1][width], -L[v][1][width]]
    added = 0
    for u in graph:
        for v in graph[u]:
            if u >= v:  # vô hướng, tránh lặp
                continue
            m.add_constraint(L_aux[(u, 1, width)] + L_aux[(v, 1, width)] <= 1,
                             ctname=f"aw_LL_{u}_{v}")
            added += 1

            # 2) phần còn lại y hệt SAT
            for label in range(2, k - width + 2):
                block_id = (label - 1) // width + 1
                if label % width == 1:
                    # [-R[u][block_id-1][width], -R[v][block_id-1][width]]
                    if block_id - 1 >= 1:
                        m.add_constraint(R_aux[(u, block_id - 1, width)] + R_aux[(v, block_id - 1, width)] <= 1,
                                         ctname=f"aw_RR_{u}_{v}_{label}")
                        added += 1
                else:
                    wmod = (label - 1) % width
                    lu = L_aux[(u, block_id, width - wmod)]
                    ru = R_aux[(u, block_id, wmod)]
                    lv = L_aux[(v, block_id, width - wmod)]
                    rv = R_aux[(v, block_id, wmod)]

                    # bốn mệnh đề:
                    m.add_constraint(lu + lv <= 1, ctname=f"aw_LL2_{u}_{v}_{label}a")
                    m.add_constraint(lu + rv <= 1, ctname=f"aw_LR2_{u}_{v}_{label}b")
                    m.add_constraint(ru + lv <= 1, ctname=f"aw_RL2_{u}_{v}_{label}c")
                    m.add_constraint(ru + rv <= 1, ctname=f"aw_RR2_{u}_{v}_{label}d")
                    added += 4

    print(f"CPLEX build done — vars={m.number_of_variables}, cons={m.number_of_constraints} (+anti={added})", flush=True)

    # Feasibility solve
    sol = m.solve(log_output=False)
    num_vars = m.number_of_variables
    num_cons = m.number_of_constraints
    verdict = sol is not None

    if queue is not None:
        queue.put(num_vars)
        queue.put(num_cons)
        queue.put(verdict)

    if verdict:
        print(f"Solution found: width={width}", flush=True)
    else:
        print("No solution exists", flush=True)

    end = time.time()
    print(f"Time taken: {round(end - start, 3)} seconds", flush=True)
    return verdict


# =========================
# Driver (giữ nguyên API cũ)
# =========================

res = [["filename", "n", "k", "proportion", "lower_bound", "upper_bound", "width", "num_vars", "num_constraints", "verdict", "time"]]
res2 = []

def read_input(file_path):
    graph = {}
    with open(file_path, 'r') as f:
        n, e = map(int, f.readline().split())
        for i in range(1, n + 1):
            graph[i] = []
        for _ in range(e):
            u, v = map(int, f.readline().split())
            graph[u].append(v)
            graph[v].append(u)  # coi như vô hướng
    return graph

def run_test_with_timeout(graph, k, width, time_left_sec=3600, threads=None):
    log_file = open("log_random.txt", "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    start = time.time()
    queue = multiprocessing.Queue()

    p = multiprocessing.Process(
        target=solve_no_hole_anti_k_labeling_cplex,
        args=(graph, k, width, queue, time_left_sec, threads)
    )
    p.start()
    p.join(timeout=time_left_sec)
    if p.is_alive():
        p.terminate()
        p.join()

    num_var = queue.get() if not queue.empty() else None
    num_cons = queue.get() if not queue.empty() else None
    verdict = queue.get() if not queue.empty() else None

    res2.extend([num_var, num_cons, verdict])
    elapsed = round((time.time() - start), 2)
    print(f"[Test k={k}, w={width}] Time: {elapsed} seconds", flush=True)
    return bool(verdict)

def binary_search_for_ans(graph, k, left, right, file, timeout_sec=1800, threads=None):
    global res, res2
    res.append([file, None, None, None, None, None, None])
    time_left = timeout_sec
    ans = -9999
    while left <= right:
        width = (left + right) // 2
        t0 = time.time()
        res2.extend([file, k, width])
        ok = run_test_with_timeout(graph, k, width, time_left_sec=time_left, threads=threads)
        res2.append(round(time.time() - t0, 2))
        res.append(res2); res2 = []
        time_left -= time.time() - t0
        if ok:
            ans = width; left = width + 1
        else:
            right = width - 1
        if time_left <= 0.5:
            return -ans if ans != -9999 else -9999
    return ans

def tuantu_for_ans(graph, k, rand, lower_bound, upper_bound, file, timeout_sec=3600, threads=None):
    global res, res2
    res.append([None, None, None, None, None, None, None, None, None, None, None])
    time_left = timeout_sec
    ans = -9999
    width = lower_bound
    while True:
        t0 = time.time()
        res2.extend([file, len(graph), k, rand, lower_bound, upper_bound, width])
        ok = run_test_with_timeout(graph, k, width, time_left_sec=time_left, threads=threads)
        res2.append(round(time.time() - t0, 2))
        res.append(res2); res2 = []
        time_left -= time.time() - t0
        if ok:
            ans = width
            width += 1
            if time_left <= 0.5 or ans == upper_bound:
                return -ans if ans != -9999 else -9999
        else:
            if time_left <= 0.5:
                return -ans if ans != -9999 else -9999
            break
    return ans

def write_to_excel(data, output_file='output/output_test_1800s_random_50_80.xlsx'):
    log_file = open("log_random.txt", "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    try:
        df = pd.DataFrame(data)
        if df.shape[1] >= 11:
            df.columns = ["filename","n","k","proportion","lower_bound","upper_bound","width",
                          "num_vars","num_constraints","verdict","time"]
        df.to_excel(output_file, index=False)
        print(f"Data written to {output_file}", flush=True)
    except Exception as e:
        print(f"Error writing to Excel: {e}", flush=True)

def cnf():  # runner
    log_file = open("log_random.txt", "w", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    folder_path = "data/11. hb"
    files = glob.glob(f"{folder_path}/*")
    lst, filename = [], []
    upper_bound = [7,9,17,9,22,13,14,8,24,36,51,39,35,102,79,220,64,256,104,220,326,136,113]
    lower_bound = [6,9,16,8,21,12,12,8,19,32,46,39,28, 91,78,219,46,256,103,219,326,136,112]
    proportion  = [77,57,56,62,72,62,56,64,79,56,69,53,77,52,75,66,69,59,64,78,60,58,70]
    for file in files:
        lst.append(folder_path + "/" + os.path.basename(file))
        filename.append(os.path.basename(file))

    threads = None  # ví dụ: 4

    for i in range(10, len(lst)):
        t0 = time.time()
        graph = read_input(lst[i])
        rand = proportion[i]
        k = len(graph) * rand // 100
        file = filename[i]
        time_limit = 3600
        # ans = binary_search_for_ans(graph, k, 2, k-1, file, time_limit, threads=threads)
        ans = tuantu_for_ans(graph, k, rand, lower_bound[i]*rand//100, upper_bound[i], file, time_limit, threads=threads)
        print("$$$$"); print(ans); print("$$$$")
        t1 = time.time()
        print(f"Time taken for {file} with k = {k}: {round(t1 - t0, 2)} seconds", flush=True)
        if ans >= 0:
            print(f"Maximum width for {file} is {ans}", flush=True)
        else:
            if ans == -9999:
                print(f"No answer before timeout for {file}", flush=True)
            else:
                print(f"Maximum width before timeout for {file} is {-ans}", flush=True)
            print("time out", flush=True)
        return

    write_to_excel(res)

if __name__ == "__main__":
    cnf()
