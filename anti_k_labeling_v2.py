import glob
import multiprocessing
import os
import random
import time
import logging
from pysat.card import CardEnc
from pysat.pb import PBEnc
from pysat.solvers import Solver
import pandas as pd

#  ./painless/build/release/painless_release cnf/K_n117_k80/K_n117_k80_w38.cnf   -c=4   -solver=cckk -no-model

# Global config
LOG_FILE = "log_random.txt"
EXCEL_FILE = "output/output_test_1800s_random_50_80.xlsx"
top_id = 2

# ----------------------------- Logging setup -----------------------------
def setup_logger(name: str = "akl", log_file: str = LOG_FILE, level=logging.INFO) -> logging.Logger:
    """
    Tạo logger ghi vào file LOG_FILE. Gọi ở mỗi process một lần.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Xóa handler cũ (nếu có) để tránh nhân đôi
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(processName)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger._configured = True  # type: ignore[attr-defined]
    return logger
# ------------------------------------------------------------------------


def solve_no_hole_anti_k_labeling(graph, k, width, queue):
    logger = setup_logger()

    if width <= 1:
        logger.error("Width must be greater than 1!!!!!!!!!!!!")
        return False

    start = time.time()
    global top_id
    top_id = 2

    solver = Solver(name='Cadical195')
    clauses = [[-1]]  # giữ nguyên như code gốc

    n = len(graph)
    block = (k - 1) // width
    last_block_size = k - (block) * width

    # x[i][label]
    x = [[1]]
    for i in range(1, n + 1):
        tmp = [1]
        for label in range(1, k + 1):
            tmp.append(top_id)
            top_id += 1
        x.append(tmp)

    # L[i][block_id][index]
    L = [[[1]]]
    for i in range(1, n + 1):
        tmp = [[1]]
        for block_id in range(1, block + 1):
            tmp2 = [1]
            if block_id == 1:
                for _ in range(1, width + 1):
                    tmp2.append(top_id)
                    top_id += 1
            else:
                for _ in range(1, width):
                    tmp2.append(top_id)
                    top_id += 1
            tmp.append(tmp2)
        L.append(tmp)

    # R[i][block_id][index]
    R = [[[1]]]
    for i in range(1, n + 1):
        tmp = [[1]]
        for block_id in range(1, block + 1):
            tmp2 = [1]
            if block_id == block:
                # Last block may not be full
                last_block_size = k - (block) * width
                for _ in range(1, last_block_size + 1):
                    tmp2.append(top_id)
                    top_id += 1
            else:
                for _ in range(1, width + 1):
                    tmp2.append(top_id)
                    top_id += 1
            tmp.append(tmp2)
        R.append(tmp)

    clauses.extend(Symetry_breaking(graph, x))

    # Encode SCL <= 1
    for i in range(1, n + 1):
        for block_id in range(1, block + 1):
            order = [-1]
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
                order = [-1]
                for index in range(1, last_block_size + 1):
                    label = block_id * width + index
                    order.append(x[i][label])
                clauses.extend(SCL_AMO(order, R, i, block_id, last_block_size))
            else:
                order = [-1]
                for index in range(1, width + 1):
                    label = block_id * width + index
                    order.append(x[i][label])
                clauses.extend(SCL_AMO(order, R, i, block_id, width))

    # Exactly one
    for i in range(1, n + 1):
        tmp = []
        tmp.append(L[i][1][width])
        for block_id in range(1, block):
            tmp.append(R[i][block_id][width])
        tmp.append(R[i][block][last_block_size])
        clauses.extend(Exactly_One1(tmp))

    # Every label must be used at least once
    for label in range(1, k + 1):
        clause = []
        for i in range(1, n + 1):
            clause.append(x[i][label])
        clauses.append(clause)

    # Hai đỉnh nối nhau không được gán nhãn có chênh lệch < width
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

    solver.append_formula(clauses)

    logger.info(f"Number of variables: {solver.nof_vars()}")
    logger.info(f"Number of clauses: {solver.nof_clauses()}")

    queue.put(solver.nof_vars())
    queue.put(solver.nof_clauses())

    # Ghi CNF ra file để chạy Painless bên ngoài nếu muốn
    write_cnf_to_file(clauses, solver, n, k, width)

    if solver.solve():
        queue.put(True)
        logger.info(f"Solution found: {width}")
        end = time.time()
        logger.info(f"Time taken: {end - start} seconds")
        return True
    else:
        queue.put(False)
        logger.info("No solution exists")
        end = time.time()
        logger.info(f"Time taken: {end - start} seconds")
        return False


def SCL_AMO(order, R, i, block_id, width):
    clauses = []
    # (9)
    for index in range(1, width + 1):
        clauses.append([-order[index], R[i][block_id][index]])
    # (10)
    for index in range(1, width):
        clauses.append([-R[i][block_id][index], R[i][block_id][index + 1]])
    # (11)
    clauses.append([order[1], -R[i][block_id][1]])
    # (12)
    for index in range(2, width + 1):
        clauses.append([order[index], R[i][block_id][index - 1], -R[i][block_id][index]])
    # (13)
    for index in range(2, width + 1):
        clauses.append([-order[index], -R[i][block_id][index - 1]])
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
    # BDD-based encoding for Exactly One
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
    global top_id
    cnf = CardEnc.equals(lits=variables, bound=1, encoding=encoding, top_id=top_id)
    top_id = cnf.nv
    return list(cnf.clauses)


def At_Most_K(variables, k, encoding=1):
    global top_id
    cnf = CardEnc.atmost(lits=variables, bound=k, encoding=encoding, top_id=top_id)
    top_id = cnf.nv
    return list(cnf.clauses)


def Exactly_One(variables):
    clauses = []
    global top_id

    # At least one
    clauses.append(variables)

    # Ladder AMO
    R = []
    for _ in range(len(variables)):
        R.append(top_id)
        top_id += 1

    for i in range(len(variables)):
        clauses.append([-variables[i], R[i]])
    for i in range(1, len(variables)):
        clauses.append([-variables[i], -R[i - 1]])
    clauses.append([variables[0], -R[0]])
    for i in range(1, len(variables)):
        clauses.append([variables[i], R[i - 1], -R[i]])
    for i in range(len(variables) - 1):
        clauses.append([-R[i], R[i + 1]])

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


def run_test_with_timeout(graph, k, width, timeout_sec=3600):
    logger = setup_logger()
    global res2

    start = time.time()
    queue = multiprocessing.Queue()

    p = multiprocessing.Process(
        target=solve_no_hole_anti_k_labeling,
        args=(graph, k, width, queue)
    )
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()

    num_var = queue.get() if not queue.empty() else None
    num_clause = queue.get() if not queue.empty() else None
    verdict = queue.get() if not queue.empty() else None
    res2.extend([num_var, num_clause, verdict])

    elapsed = round((time.time() - start), 2)

    if verdict is True:
        logger.info(f"Found a solution {width}")
        logger.info(f"[Test k={k}, w={width}] Time: {elapsed} seconds")
        return True
    else:
        logger.info(f"No solution found {width}")
        logger.info(f"[Test k={k}, w={width}] Time: {elapsed} seconds")
        return False


res = [["filename", "n", "k", "proportion", "lower_bound",
        "upper_bound", "width", "num_vars", "num_clauses", "verdict", "time"]]
res2 = []


def tuantu_for_ans(graph, k, rand, lower_bound, upper_bound, file, timeout_sec=3600):
    logger = setup_logger()
    global res, res2

    res.append([None, None, None, None, None, None, None, None, None, None, None])
    time_left = timeout_sec
    ans = -9999
    width = lower_bound

    while True:
        time_start = time.time()
        res2.extend([file, len(graph), k, rand, lower_bound, upper_bound, width])

        if run_test_with_timeout(graph, k, width, time_left):
            res2.append(round(time.time() - time_start, 2))
            res.append(res2)
            res2 = []

            time_left -= time.time() - time_start
            ans = width
            width += 1

            if time_left <= 0.5 or ans == upper_bound:
                if ans == -9999:
                    return -9999
                return -ans
        else:
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


def write_to_excel(data, output_file=EXCEL_FILE):
    logger = setup_logger()
    try:
        df = pd.DataFrame(data)
        df.to_excel(output_file, index=False)
        logger.info(f"Data written to {output_file}")
    except Exception:
        logger.exception("Error writing to Excel")


def solve():
    logger = setup_logger()
    # clear file at start
    with open(LOG_FILE, "w", encoding="utf-8"):
        pass

    logger.info("=== Start solve() ===")

    folder_path = "data/11. hb"
    files = glob.glob(f"{folder_path}/*")

    lst = []
    filename = []

    upper_bound = [7, 9, 17, 9, 22, 13, 14, 8, 24, 36, 51, 39, 35, 102, 79, 220, 64, 256, 104, 220, 326, 136, 113]
    lower_bound = [6, 9, 16, 8, 21, 12, 12, 8, 19, 32, 46, 39, 28, 91, 78, 219, 46, 256, 103, 219, 326, 136, 112]
    proportion = [77, 57, 56, 62, 72, 62, 56, 64, 79, 56, 69, 53, 77, 52, 75, 66, 69, 59, 64, 78, 60, 58, 70]

    for file in files:
        lst.append(os.path.join(folder_path, os.path.basename(file)))
        filename.append(os.path.basename(file))

    for i in range(10, len(lst)):
        time_start = time.time()
        graph = read_input(lst[i])
        rand = proportion[i]
        k = len(graph) * rand // 100
        file = filename[i]
        ans = -9999
        time_limit = 1800

        ans = tuantu_for_ans(graph, k, rand, lower_bound[i] * rand // 100, upper_bound[i], file, time_limit)

        logger.info("$$$$")
        logger.info(str(ans))
        logger.info("$$$$")

        time_end = time.time()
        logger.info(f"Time taken for {file} with k = {k}: {time_end - time_start} seconds")

        if ans >= 0:
            logger.info(f"Maximum width for {file} is {ans}")
        else:
            if ans == -9999:
                logger.info(f"No answer before timeout for {file}")
            else:
                logger.info(f"Maximum width before timeout for {file} is {-ans}")
            logger.info("time out")
        return

    write_to_excel(res)


def write_cnf_to_file(clauses, solver, n, k, width):
    """
    Write SAT solver clauses to a CNF file in DIMACS format.
    """
    logger = setup_logger()
    base_path = "cnf"
    folder_path = os.path.join(base_path, f"K_n{n}_k{k}")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cnf_filename = os.path.join(folder_path, f"K_n{n}_k{k}_w{width}.cnf")

    try:
        with open(cnf_filename, "w", encoding="utf-8") as f:
            f.write(f"p cnf {solver.nof_vars()} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        logger.info(f"CNF written to {cnf_filename}")
    except Exception:
        logger.exception("Failed to write CNF")


if __name__ == "__main__":
    solve()
