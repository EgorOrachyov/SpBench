# Clear stats from previous runs
for i in Summary-*; do
  if [[ -f $i ]]; then
    echo "Remove file: $i"
    rm $i
  fi
done