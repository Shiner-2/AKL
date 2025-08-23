#!/bin/bash
# ./painless/build/release/painless_release cnf/ibm32.mtx.rnd_n32_k25/ibm32.mtx.rnd_n32_k25_w2.cnf -c=2 -solver=ck -no-model
TIME_LIMIT=600
OUTPUT_DIR="res"
mkdir -p "$OUTPUT_DIR"
OUTPUT_CSV="$OUTPUT_DIR/results_fixed_$(date +"%Y%m%d_%H%M%S").csv"
echo "File,Result,Variables,Clauses,Time(seconds),Memory(kB)" > "$OUTPUT_CSV"

process_cnf() {
    local file="$1"
    echo "Processing $file..."
    
    # Đọc metadata CNF
    local header=$(head -n 100 "$file" | grep -m 1 "^p cnf")
    local vars=$(echo "$header" | awk '{print $3}')
    local clauses=$(echo "$header" | awk '{print $4}' | tr -d '\r\n')
    
    exec 3>&1
    local output=$( { timeout $TIME_LIMIT /usr/bin/time -f "%e,%M" ./painless/build/release/painless_release "$file" -c=8 -solver=cccckkkk -no-model 2>&1 1>&3; } 3>&1 )
    local exit_status=$?
    exec 3>&-
    
    local result=$(echo "$output" | grep -E "^s (SATISFIABLE|UNSATISFIABLE)" | awk '{print $2}')
    local time_mem=$(echo "$output" | grep -E "^[0-9]+.[0-9]+,[0-9]+$" | tail -n 1)
    local time=$(echo "$time_mem" | cut -d',' -f1)
    local mem=$(echo "$time_mem" | cut -d',' -f2)
    
    if [[ $exit_status -eq 124 ]]; then
        result="TIMEOUT"
        time="$TIME_LIMIT"
        mem="0"
    fi
    
    echo "$file,${result:-N/A},${vars:-0},${clauses:-0},${time:-0},${mem:-0}" >> "$OUTPUT_CSV"
    
    [[ "$result" == "UNSATISFIABLE" ]] && return 1 || return 0
}

# Duyệt theo thứ tự tự nhiên (natural sort)
while IFS= read -r -d '' folder; do
    if [ -d "$folder" ]; then
        echo "=== Processing folder: $folder ==="
        
        # Sắp xếp file theo thứ tự tự nhiên
        while IFS= read -r -d '' file; do
            process_cnf "$file"
            [[ $? -eq 1 ]] && { echo "UNSAT found, skipping folder"; break; }
        done < <(find "$folder" -maxdepth 1 -name "*.cnf" -print0 | sort -z -V)
    fi
done < <(find cnf -type d -print0 | sort -z -V)

echo "Done! Results saved to $OUTPUT_CSV"