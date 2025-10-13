from multiprocessing import Process, Queue
from docplex.mp.model import Model
import time, os, pandas as pd

# ============================================================
# 💡 CONFIG
# ============================================================
INPUT_FOLDER = "data/11. hb"
TIME_LIMIT = 1800
LOG_FILE = "log_k_labeling.txt"
EXCEL_FILE = "results_k_labeling.xlsx"

upper_bound = [7,9,17,9,22,13,14,8,24,36,51,39,35,102,79,220,64,256,104,220,326,136,113]
lower_bound = [6,9,16,8,21,12,12,8,19,32,46,39,28,91,78,219,46,256,103,219,326,136,112]
proportion = [77,57,56,62,72,62,56,64,79,56,69,53,77,52,75,66,69,59,64,78,60,58,70]

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
# 👷 WORKER: chạy 1 lần solve trong sub-process
# ============================================================
def _worker_solve(n, E, k, w, time_lim, q):
    """
    Worker chạy trong process con: build + solve model.
    Trả kết quả qua Queue dưới dạng dict đơn giản.
    """
    t0 = time.time()
    try:
        m = Model(name=f"AntiKLabel_k{k}_w{w}", log_output=False)

        V = range(1, n + 1)
        L = range(1, w + 1)
        x = {(v, l): m.binary_var(name=f"x_{v}_{l}") for v in V for l in L}

        # Mỗi đỉnh đúng 1 nhãn
        for v in V:
            m.add_constraint(m.sum(x[v, l] for l in L) == 1)

        # Mỗi nhãn được dùng ít nhất 1 lần
        for l in L:
            m.add_constraint(m.sum(x[v, l] for v in V) >= 1)

        # Ràng buộc anti-k: các đỉnh kề nhau có nhãn cách nhau ≥ k
        for (u, v) in E:
            for l1 in range(1, w + 1):
                lo = max(1, l1 - (k - 1))
                hi = min(w, l1 + (k - 1))
                for l2 in range(lo, hi + 1):
                    m.add_constraint(x[u, l1] + x[v, l2] <= 1)

        # Symmetry breaking
        for l in range(1, w // 2 + 1):
            m.add_constraint(x[1, l] == 0, f"fix_1_{l}")

        # Chỉ kiểm tra feasibility (nhẹ hơn maximize 0)
        m.minimize(0)

        # Cấu hình CPLEX
        if time_lim is None or time_lim <= 0:
            time_lim = 1e-3
        m.parameters.timelimit = float(max(1e-3, time_lim))
        m.parameters.threads = 1
        # m.parameters.emphasis.mip = 1              # (tùy chọn) ưu tiên khả thi
        # m.parameters.mip.tolerances.mipgap = 0.2   # (tùy chọn) thoát sớm
        m.parameters.workmem = 8192

        sol = m.solve(log_output=False)
        solve_time = time.time() - t0

        status = str(m.solve_details.status)
        if "time" in status.lower():
            state = "timeout"
        elif "infeasible" in status.lower():
            state = "infeasible"
        elif sol is None:
            state = "unknown"
        else:
            state = "feasible"

        assignments = {}
        if state == "feasible":
            assignments = {v: next(l for l in L if sol.get_value(x[v, l]) > 0.5) for v in V}

        q.put({
            "k": k,
            "time": solve_time,
            "status": state,
            "assignments": assignments
        })
    except Exception as e:
        # Phòng sự cố bất ngờ trong worker
        q.put({
            "k": k,
            "time": time.time() - t0,
            "status": f"unknown",
            "assignments": {},
            "error": repr(e)
        })

def solve_anti_k_labeling_subproc(n, E, k, w, wall_limit):
    """
    Chạy 1 lần solve trong sub-process và cắt đúng wall-clock theo wall_limit (giây).
    Trả về dict: {k, time, status, assignments}
    """
    if wall_limit is None or wall_limit <= 0:
        return {"k": k, "time": 0.0, "status": "timeout", "assignments": {}}

    q = Queue()
    p = Process(target=_worker_solve, args=(n, E, k, w, wall_limit, q), daemon=True)
    t0 = time.time()
    p.start()
    p.join(timeout=wall_limit)

    if p.is_alive():
        # quá hạn ⇒ kill cứng process con
        p.terminate()
        p.join()
        return {"k": k, "time": wall_limit, "status": "timeout", "assignments": {}}

    # Process kết thúc trong hạn ⇒ lấy kết quả
    try:
        result = q.get_nowait()
        # Phòng trường hợp worker gặp lỗi mà vẫn trả về dict có 'error'
        if isinstance(result, dict) and result.get("status") == "unknown" and "error" in result:
            # vẫn trả về unknown, ghi time thực tế
            result["time"] = time.time() - t0
        return result
    except Exception:
        # Không lấy được kết quả (hiếm): đánh dấu unknown
        return {"k": k, "time": time.time() - t0, "status": "unknown", "assignments": {}}

# ============================================================
# 🔍 KIỂM TRA NGHIỆM
# ============================================================
def check_solution(n, E, w, k, assignments, log):
    """
    Kiểm tra tính hợp lệ của nghiệm cho bài Anti-k-Labeling.
    """
    errors = []

    # (A) Mỗi đỉnh có đúng 1 nhãn
    if len(assignments) != n:
        errors.append(f"Số lượng đỉnh gán nhãn ({len(assignments)}) ≠ n ({n})")

    # (B) Phạm vi nhãn
    for v, lbl in assignments.items():
        if not (1 <= lbl <= w):
            errors.append(f"Đỉnh {v} có nhãn {lbl} không nằm trong [1..{w}]")

    # (C) Mọi nhãn được dùng ít nhất 1 lần
    used_labels = set(assignments.values())
    if len(used_labels) < w:
        missing = set(range(1, w + 1)) - used_labels
        errors.append(f"Các nhãn chưa dùng: {sorted(missing)}")

    # (D) Ràng buộc Anti-k
    for (u, v) in E:
        if u in assignments and v in assignments:
            if abs(assignments[u] - assignments[v]) < k:
                errors.append(f"Vi phạm khoảng cách k giữa ({u},{v}): |{assignments[u]} - {assignments[v]}| = {abs(assignments[u]-assignments[v])}")

    # Kết luận
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
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write(f"===== Anti-k-Labeling Solver (sub-process wall-clock capped) =====\n")
        log.write(f"Folder: {INPUT_FOLDER}\n")
        log.write("==========================================\n\n")
        log.flush()

        i = -1
        for filename in sorted(os.listdir(INPUT_FOLDER)):
            i += 1
            if i < 13: continue
            # if not filename.startswith("M"):
            #     continue

            filepath = os.path.join(INPUT_FOLDER, filename)
            n, E = read_input(filepath)
            LB = lower_bound[i] * proportion[i] // 100
            UB = upper_bound[i]
            w = proportion[i] * n // 100

            log.write(f"▶ File: {filename}\n")
            log.write(f"   n = {n}, |E| = {len(E)}, LB = {LB}, UB = {UB}, w = {w}\n")
            log.flush()

            total_time = TIME_LIMIT
            for k in range(LB, UB + 1):
                # Nếu gần hết giờ thì dừng sớm trước khi build/solve
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
                    "Time (s)": round(result["time"], 3)
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

                # if total_time <= 10:
                #     log.write(f"   - Not enough time left ({total_time:.2f}s), stop for this instance.\n")
                #     log.flush()
                #     break

                log.flush()

            results_list.append({})
            log.write("------------------------------------------\n\n")

        log.write("All tasks completed.\n")

    df = pd.DataFrame(results_list)
    df.to_excel(EXCEL_FILE, index=False)
    print(f"✅ Saved log to {LOG_FILE}")
    print(f"✅ Saved results to {EXCEL_FILE} ({len(df)} rows)")

# ============================================================
# 🔓 ENTRY POINT (cần để multiprocessing chạy ổn trên Windows)
# ============================================================
if __name__ == "__main__":
    main()
