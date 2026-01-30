def convert_to_opl_dat(input_file, output_file):
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # first line
    n, e, k, lb, ub = map(int, lines[0].split())

    edges = []
    for line in lines[1:]:
        u, v = map(int, line.split())
        edges.append((u, v))

    with open(output_file, "w") as f:
        f.write(f"n  = {n};\n")
        f.write(f"k  = {k};\n")
        f.write(f"lb = {lb};\n")
        f.write(f"ub = {ub};\n\n")

        f.write("E = {\n")
        for i, (u, v) in enumerate(edges):
            comma = "," if i < len(edges) - 1 else ""
            f.write(f"  <{u},{v}>{comma}\n")
        f.write("};\n")


# ===== usage =====
convert_to_opl_dat(
    "data\hb\K-dwt__234.mtx.rnd",        # file data thô
    "antiklabel.dat"    # file cho CPLEX OPL
)
