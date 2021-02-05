profilingFile="Profiling-Time.txt"

echo "Run memory profiling"
echo "Save results into $profilingFile"

if [[ -f $profilingFile ]]; then
  rm $profilingFile
fi

# For each target we run as separate process for each matrix
cat data/targets_all.txt | while read target; do
  cat data/config_profiling.txt | while read test; do

    # Ignore lines, which start from comment mark
    # Ignore falling benchmarks
    if [[ ${test::1} != "%" && ! ( $target == *"cusparse"* && $test == *"wiki-Talk.mtx"* ) ]]; then

      # Before profiling, add the target and test name
      # Then for begin and end capture time points in order to
      # later synchronize with nvidia-smi log output

      echo "Exec command: ./$target -E $test"

      echo "$target $test" >> $profilingFile
      date +"%Y/%m/%d %H:%M:%S.%3N" >> $profilingFile
      ./$target -E $test
      date +"%Y/%m/%d %H:%M:%S.%3N" >> $profilingFile
    fi
  done
done