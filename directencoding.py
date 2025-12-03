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
import subprocess
import shutil
from typing import Optional, Dict, Any


#  ./painless/build/release/painless_release cnf/K_n117_k80/K_n117_k80_w38.cnf   -c=4   -solver=cckk -no-model

# Global config
LOG_FILE = "logs/log_directencoding_painless.txt"
EXCEL_FILE = "output/output_directencoding_painless.xlsx"

# --- Painless runner config ---
PAINLESS_BIN = "./painless/build/release/painless_release"  # đổi nếu khác
PAINLESS_ARGS = ["-c=4", "-solver=cckk", "-no-model"]       # tuỳ chọn
RUN_PAINLESS = True                                         # bật/tắt chạy Painless

top_id = 2

# ----------------------------- Logging setup -----------------------------
RESULT_LEVEL_NUM = 25  # giữa INFO(20) và WARNING(30)
logging.addLevelName(RESULT_LEVEL_NUM, "RESULT")

def result(self, msg, *args, **kwargs):
    if self.isEnabledFor(RESULT_LEVEL_NUM):
        self._log(RESULT_LEVEL_NUM, msg, args, **kwargs)

logging.Logger.result = result
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


def solve_no_hole_anti_k_labeling(graph, k, width, queue, timeout_sec=3600, instance_name="A" ):
    logger = setup_logger()

    if width <= 1:
        logger.error("Width must be greater than 1!!!!!!!!!!!!")
        return False

    start = time.time()
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


    for u in graph:
        for v in graph[u]:
            
            for labelu in range(1, k + 1):
                for labelv in range(1, k + 1):
                    if abs(labelu - labelv) < width:
                        clauses.append([-x[u][labelu], -x[v][labelv]])
            # for labelu in range(1, k + 1):
            #     minv = max(0, labelu - width)
            #     maxv = min(k + 1, labelu + width)
            #     if minv == 0:
            #         clauses.append([-x[u][labelu], R[v][k - labelu - width + 1]])
            #         clauses.append([-x[u][labelu], -L[v][labelu + width - 1]])
            #     if maxv == k + 1:
            #         clauses.append([-x[u][labelu], L[v][labelu - width]])
            #         clauses.append([-x[u][labelu], -R[v][k - labelu + width]])
            #     if minv > 0 and maxv < k + 1:
            #         clauses.append([-x[u][labelu], L[v][labelu - width], R[v][k - labelu - width + 1]])


            # for labelv in range(1, k + 1):
            #     minu = max(0, labelv - width)
            #     maxu = min(k + 1, labelv + width)
            #     if minu == 0:
            #         clauses.append([-x[v][labelv], R[u][k - labelv - width + 1]])
            #         clauses.append([-x[v][labelv], -L[u][labelv + width - 1]])
            #     if maxu == k + 1:
            #         clauses.append([-x[v][labelv], L[u][labelv - width]])
            #         clauses.append([-x[v][labelv], -R[u][k - labelv + width]])
            #     if minu > 0 and maxu < k + 1:
            #         clauses.append([-x[v][labelv], L[u][labelv - width], R[u][k - labelv - width + 1]])


    solver.append_formula(clauses)

    logger.info(f"Number of variables: {solver.nof_vars()}")
    logger.info(f"Number of clauses: {solver.nof_clauses()}")

    queue.put(solver.nof_vars())
    queue.put(solver.nof_clauses())

    # Ghi CNF ra file để chạy Painless bên ngoài nếu muốn
    cnf_path = write_cnf_to_file(clauses, solver, n, k, width, instance_name)

    # Gọi Painless (nếu bật) và quyết định ngay theo kết quả
    if RUN_PAINLESS and cnf_path:
        # Dùng timeout nếu có biến timeout_sec trong scope, không thì mặc định 1800
        try:
            painless_timeout = min(1800, timeout_sec)  # timeout_sec có ở caller? nếu không sẽ rơi vào except
        except NameError:
            painless_timeout = 1800

        pres = run_painless(cnf_path, timeout_sec=painless_timeout)  # đổi timeout nếu cần
        logger.result(f"[PAINLESS] result: status={pres['status']} time={pres['time_sec']:.2f}s rc={pres['returncode']}")

        if pres["status"] == "SAT":
            queue.put(True)
            logger.result(f"[PAINLESS] Shortcut: SAT at width={width}")
            end = time.time()
            logger.result(f"Time taken: {end - start} seconds")
            return True
        elif pres["status"] == "UNSAT":
            queue.put(False)
            logger.result(f"[PAINLESS] Shortcut: UNSAT at width={width}")
            end = time.time()
            logger.result(f"Time taken: {end - start} seconds")
            return False
        # Các trạng thái khác (TIMEOUT/ERROR/UNKNOWN) thì rơi xuống PySAT

    logger.error("Some thing wrong with PAINLESS, Solving with PySAT...")
    # Fallback: giải bằng PySAT nếu Painless không kết luận được
    if solver.solve():
        queue.put(True)
        logger.result(f"Solution found: {width}")
        end = time.time()
        logger.result(f"Time taken: {end - start} seconds")
        return True
    else:
        queue.put(False)
        logger.result("No solution exists")
        end = time.time()
        logger.result(f"Time taken: {end - start} seconds")
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


def run_test_with_timeout(graph, k, width, timeout_sec=3600, instance_name="A"):
    logger = setup_logger()
    global res2

    start = time.time()
    queue = multiprocessing.Queue()

    p = multiprocessing.Process(
        target=solve_no_hole_anti_k_labeling,
        args=(graph, k, width, queue, timeout_sec, instance_name)
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

        if run_test_with_timeout(graph, k, width, time_left, file[0]):
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

def run_painless(cnf_path: str,
                 bin_path: str = PAINLESS_BIN,
                 extra_args=None,
                 timeout_sec: int = 1800,
                 cwd: Optional[str] = None) -> Dict[str, Any]:
    """
    Gọi Painless trên file CNF và trả kết quả:
      {
        "status": "SAT"|"UNSAT"|"UNKNOWN"|"TIMEOUT"|"ERROR",
        "time_sec": float,
        "returncode": int | None,
        "stdout": str,
        "stderr": str,
        "cmd": [..]
      }
    """
    logger = setup_logger()
    if extra_args is None:
        extra_args = PAINLESS_ARGS

    # Kiểm tra binary tồn tại/khả dụng
    exists = os.path.exists(bin_path)
    in_path = shutil.which(bin_path) is not None
    if not exists and not in_path:
        logger.error(f"Painless binary not found: {bin_path}")
        return {"status": "ERROR", "time_sec": 0.0, "returncode": None,
                "stdout": "", "stderr": f"Binary not found: {bin_path}", "cmd": []}

    cmd = [bin_path] + list(extra_args) + [cnf_path]
    t0 = time.time()
    logger.info(f"[PAINLESS] Running: {' '.join(cmd)}")

    try:
        cp = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        dt = time.time() - t0
        out, err = cp.stdout, cp.stderr
        rc = cp.returncode

        # Parse trạng thái theo chuẩn DIMACS
        if rc == 10 or "s SATISFIABLE" in out:
            status = "SAT"
        elif rc == 20 or "s UNSATISFIABLE" in out:
            status = "UNSAT"
        elif rc == 0:
            status = "UNKNOWN"  # chạy ok nhưng không in kết luận chuẩn
        else:
            status = "ERROR"

        logger.info(f"[PAINLESS] status={status} rc={rc} time={dt:.2f}s")
        if err.strip():
            logger.info(f"[PAINLESS][stderr] {err.strip().splitlines()[-1]}")

        return {"status": status, "time_sec": dt, "returncode": rc,
                "stdout": out, "stderr": err, "cmd": cmd}

    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        logger.info(f"[PAINLESS] TIMEOUT after {dt:.2f}s")
        return {"status": "TIMEOUT", "time_sec": dt, "returncode": None,
                "stdout": "", "stderr": "", "cmd": cmd}
    except Exception as ex:
        dt = time.time() - t0
        logger.exception("[PAINLESS] Exception while running solver")
        return {"status": "ERROR", "time_sec": dt, "returncode": None,
                "stdout": "", "stderr": str(ex), "cmd": cmd}


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

    for i in range(0, len(lst)):
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
        
        
        
    write_to_excel(res)


def write_cnf_to_file(clauses, solver, n, k, width, instance_name="A"):
    """
    Write SAT solver clauses to a CNF file in DIMACS format.
    Return: đường dẫn file CNF đã ghi.
    """
    logger = setup_logger()
    base_path = "cnf/v2"
    folder_path = os.path.join(base_path, f"{instance_name}_n{n}_k{k}")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cnf_filename = os.path.join(folder_path, f"{instance_name}_n{n}_k{k}_w{width}.cnf")

    try:
        with open(cnf_filename, "w", encoding="utf-8") as f:
            f.write(f"p cnf {solver.nof_vars()} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        logger.info(f"CNF written to {cnf_filename}")
        return cnf_filename
    except Exception:
        logger.exception("Failed to write CNF")
        return None

def write_to_excel(data, output_file=EXCEL_FILE):
    logger = setup_logger()
    try:
        df = pd.DataFrame(data)
        df.to_excel(output_file, index=False)
        logger.info(f"Data written to {output_file}")
    except Exception:
        logger.exception("Error writing to Excel")


if __name__ == "__main__":
    solve()
