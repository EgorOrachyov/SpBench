echo "Run benchmark for targets: $SPBENCH_TARGETS"

# Clear stats from previous runs
for i in Summary*; do
  if [[ -f $i ]]; then
    echo "Remove file: $i"
    rm $i
  fi
done

# For each target we run as separate process for each matrix
cat $SPBENCH_TARGETS | while read target; do
  cat data/config.txt | while read test; do
    # Ignore lines, which start from comment mark
    if [[ ${test::1} != "%" ]]; then
      echo "Exec command: ./$target -E $test"
      ./$target -E $test
    fi
  done
done