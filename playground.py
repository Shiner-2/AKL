def read_input(file_path):
    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()
    n, m = map(int, lines[0].split())
    edges = [tuple(map(int, line.split())) for line in lines[1:]]
    return n, edges

n, e = read_input("data/11. hb/W-685_bus.mtx.rnd")

for edge in e:
    i, j = edge
    print(f"<{i},{j}>", end = ",")