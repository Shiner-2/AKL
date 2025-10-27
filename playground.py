# -*- coding: utf-8 -*-
from multiprocessing import Process, Queue
from docplex.cp.model import CpoModel
from docplex.cp.config import context
import time, os, pandas as pd, sys

# ============================================================
# 💡 CONFIG
# ============================================================
INPUT_FOLDER = "data/11. hb"
TIME_LIMIT = 1800
LOG_FILE = "logs/log_cp_cplex.txt"
EXCEL_FILE = "res/results_cp_cplex.xlsx"

upper_bound = [7,9,17,9,22,13,14,8,24,36,51,39,35,102,79,220,64,256,104,220,326,136,113]
lower_bound = [6,9,16,8,21,12,12,8,19,32,46,39,28,91,78,219,46,256,103,219,326,136,112]
proportion  = [77,57,56,62,72,62,56,64,79,56,69,53,77,52,75,66,69,59,64,78,60,58,70]

# === Đường dẫn CP Optimizer (CHỈNH LẠI CHO ĐÚNG BẢN CỦA BẠN) ===
CPLEX_STUDIO = r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211"
CP_EXE       = os.path.join(CPLEX_STUDIO, r"cpoptimizer\bin\x64_win64\cpoptimizer.exe")
CP_DLL_DIR   = os.path.join(CPLEX_STUDIO, r"cpoptimizer\bin\x64_win64")

# ============================================================
# 🧩 ĐỌC DỮ LIỆU
# ============================================================
def read_input(file_path):
    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()
    n, m = map(int, lines[0].split())
    edges = [tuple(map(int, line.split())) for line in lines[1:]]
    return n, edges

# ============================================================
# 👷 WORKER (CP)
# ============================================================
def _worker_solve_cp(n, E, k, w, time_lim, q):
    """
    Worker CP: build + solve model CP.
    Trả kết quả qua Queue: {k, time, status, assignments, error?}
    """
    t0 = time.time()
    # Ghi log ra stderr nếu cần:
    # sys.stderr.write(f"Child PID={os.getpid()} start\n"); sys.stderr.flush()
    try:
        # --- Bắt buộc: add DLL dir (Windows, Python >=3.8) ---
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(CP_DLL_DIR)
            except Exception:
                pass

        # --- Chỉ định execfile cho CP Optimizer ---
        context.solver.local.execfile = CP_EXE

        # Guard nhanh: nếu có cạnh và k > w-1 => infeasible ngay
        if any(True for _ in E) and k > max(0, w - 1):
            q.put({"k": k, "time": time.time() - t0, "status": "infeasible", "assignments": {},
                   "error": "k > w-1 with edges ⇒ infeasible"})
            return

        # CP model
        m = CpoModel(name=f"AKL_CP_k{k}_w{w}")

        # Biến nhãn 1..w cho mỗi đỉnh
        V = list(range(1, n + 1))
        if w <= 0:
            q.put({"k": k, "time": time.time() - t0, "status": "infeasible", "assignments": {},
                   "error": "w <= 0"})
            return
        lbl = {v: m.integer_var(1, w, name=f"lbl_{v}") for v in V}

        # Tập cạnh vô hướng (tránh trùng)
        undirected = set()
        for (u, v) in E:
            if u < v: undirected.add((u, v))
            elif v < u: undirected.add((v, u))

        # Ràng buộc anti-k
        for (u, v) in undirected:
            m.add(abs(lbl[u] - lbl[v]) >= k)

        # No-hole: mỗi nhãn phải được dùng >= 1 lần
        lbl_list = [lbl[v] for v in V]
        # Nếu w > n thì infeasible (không thể dùng ≥1 cho >n nhãn)
        if w > n:
            q.put({"k": k, "time": time.time() - t0, "status": "infeasible", "assignments": {},
                   "error": "w > n ⇒ cannot use each label at least once"})
            return
        for l in range(1, w + 1):
            m.add(m.count(lbl_list, l) >= 1)

        # Tham số solver
        if time_lim is None or time_lim <= 0:
            time_lim = 1e-3

        # Bạn có thể tăng Workers nếu muốn:
        res = m.solve(TimeLimit=float(max(1e-3, time_lim)),
                      Workers=1,
                      LogVerbosity='Terse')  # để dễ xem log khi debug
        solve_time = time.time() - t0

        # Mapping trạng thái chi tiết
        state = "unknown"
        try:
            if res is not None and res.is_solution():
                state = "feasible"
            else:
                sst = str(res.get_solve_status()) if res is not None else ""
                # Một số trạng thái phổ biến:
                #  - 'Optimal', 'Feasible', 'FeasibleWithWarning'
                #  - 'Infeasible', 'InfeasibleOrUnbounded'
                #  - 'Unknown', 'JobFailed', 'SearchStopped'
                up = sst.upper()
                if "INFEAS" in up:
                    state = "infeasible"
                elif "OPTIMAL" in up or "FEASIBLE" in up:
                    state = "feasible"
                elif "UNKNOWN" in up or "JOBFAILED" in up or "SEARCHSTOPPED" in up:
                    # nếu chạm time limit gần đúng thì xem như timeout
                    state = "timeout" if solve_time + 1e-6 >= float(time_lim) else "unknown"
                else:
                    state = "timeout" if solve_time + 1e-6 >= float(time_lim) else "unknown"
        except Exception as e:
            state = "unknown"

        assignments = {}
        if state == "feasible":
            try:
                assignments = {v: res[lbl[v]] for v in V}
            except Exception:
                assignments = {}

        q.put({
            "k": k,
            "time": solve_time,
            "status": state,
            "assignments": assignments,
            "solver_status": (str(res.get_solve_status()) if res is not None else "None")
        })

    except Exception as e:
        q.put({
            "k": k,
            "time": time.time() - t0,
            "status": "unknown",
            "assignments": {},
            "error": repr(e)
        })

def solve_anti_k_labeling_subproc(n, E, k, w, wall_limit):
    if wall_limit is None or wall_limit <= 0:
        return {"k": k, "time": 0.0, "status": "timeout", "assignments": {}}

    q = Queue()
    p = Process(target=_worker_solve_cp, args=(n, E, k, w, wall_limit, q), daemon=True)
    t0 = time.time()
    p.start()
    p.join(timeout=wall_limit)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"k": k, "time": wall_limit, "status": "timeout", "assignments": {}}

    try:
        result = q.get_nowait()
        # nếu có lỗi, in ra để bạn thấy nguyên nhân thật
        if "error" in result:
            print(f"[CP child error] k={k}: {result['error']}")
        return result
    except Exception:
        return {"k": k, "time": time.time() - t0, "status": "unknown", "assignments": {}}

# ============================================================
# 🔍 KIỂM TRA NGHIỆM
# ============================================================
def check_solution(n, E, w, k, assignments, log):
    errors = []
    if len(assignments) != n:
        errors.append(f"Số lượng đỉnh gán nhãn ({len(assignments)}) ≠ n ({n})")
    for v, lbl in assignments.items():
        if not (1 <= lbl <= w):
            errors.append(f"Đỉnh {v} có nhãn {lbl} không nằm trong [1..{w}]")
    used_labels = set(assignments.values())
    if len(used_labels) < w:
        missing = set(range(1, w + 1)) - used_labels
        errors.append(f"Các nhãn chưa dùng: {sorted(missing)}")
    for (u, v) in E:
        if u in assignments and v in assignments:
            if abs(assignments[u] - assignments[v]) < k:
                errors.append(f"Vi phạm khoảng cách k giữa ({u},{v}): |{assignments[u]} - {assignments[v]}| = {abs(assignments[u]-assignments[v])}")
    if errors:
        log.write("❌ Nghiệm KHÔNG hợp lệ:\n")
        for e in errors:
            log.write(f"  - {e}\n")
        log.flush()
        return False
    else:
        log.write("✅ Nghiệm hợp lệ cho Anti-k-Labeling.\n")
        log.flush()
        return True

# ============================================================
# 🚀 CHƯƠNG TRÌNH CHÍNH
# ============================================================
def main():
    results_list = []
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(EXCEL_FILE), exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write(f"===== Anti-k-Labeling (CP Optimizer, sub-process capped) =====\n")
        log.write(f"Folder: {INPUT_FOLDER}\n")
        log.write("==========================================\n\n")
        log.flush()

        i = -1
        for filename in sorted(os.listdir(INPUT_FOLDER)):
            i += 1
            if i > 3:
                continue

            filepath = os.path.join(INPUT_FOLDER, filename)
            n, E = read_input(filepath)
            LB = lower_bound[i] * proportion[i] // 100
            UB = upper_bound[i]
            w = max(1, proportion[i] * n // 100)

            log.write(f"▶ File: {filename}\n")
            log.write(f"   n = {n}, |E| = {len(E)}, LB = {LB}, UB = {UB}, w = {w}\n")
            log.flush()

            total_time = TIME_LIMIT
            for k in range(LB, UB + 1):
                if total_time <= 1.0:
                    log.write(f"   - Not enough time left ({total_time:.2f}s), stop for this instance.\n")
                    log.flush()
                    break

                log.write(f"   - Running k = {k} ... ")
                log.flush()

                result = solve_anti_k_labeling_subproc(n, E, k, w, total_time)
                total_time -= result["time"]

                results_list.append({
                    "Filename": filename,
                    "n": n,
                    "Edges": len(E),
                    "LB": LB,
                    "UB": UB,
                    "w": w,
                    "k": result["k"],
                    "Status": result["status"],
                    "Time (s)": round(result["time"], 3),
                    "SolveStatus": result.get("solver_status", "")
                })

                if result["status"] == "feasible":
                    log.write(f"Feasible ✓ (time = {result['time']:.2f}s)\n")
                    check_solution(n, E, w, k, result["assignments"], log)
                elif result["status"] == "timeout":
                    log.write(f"Timeout ⏱ (time = {result['time']:.2f}s)\n")
                    break
                elif result["status"] == "infeasible":
                    log.write(f"Infeasible ✗ (time = {result['time']:.2f}s)\n")
                    break
                else:
                    log.write(f"Unknown ? (time = {result['time']:.2f}s)\n")
                    if "error" in result:
                        log.write(f"   Error: {result['error']}\n")
                log.flush()

            results_list.append({})
            log.write("------------------------------------------\n\n")

        log.write("All tasks completed.\n")

    df = pd.DataFrame(results_list)
    df.to_excel(EXCEL_FILE, index=False)
    print(f"✅ Saved log to {LOG_FILE}")
    print(f"✅ Saved results to {EXCEL_FILE} ({len(df)} rows)")

# ============================================================
# 🔓 ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
