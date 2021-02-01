filename="Summarize.txt"

if [[ -f $filename ]]; then
  rm $filename
fi

touch $filename

for file in Summary-*; do
  if [[ -f $file ]]; then
    cat $file >> $filename
  fi
done