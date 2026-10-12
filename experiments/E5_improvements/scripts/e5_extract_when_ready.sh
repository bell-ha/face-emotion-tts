#!/bin/zsh
# extract actors 09-24 as soon as their 60 AV clips are present (download runs in parallel)
cd "$(dirname $0)/../../.."
while pgrep -f "e5_extract.py Actor_01" >/dev/null; do sleep 20; done
todo=(09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)
while (( ${#todo} > 0 )); do
  for a in $todo; do
    if [ $(ls data/ravdess/videos/Actor_$a/01-*.mp4 2>/dev/null | wc -l) -ge 60 ]; then
      .venv/bin/python experiments/E5_improvements/scripts/e5_extract.py Actor_$a 2>&1 | grep -v -iE "warn|W0000|I0000"
      todo=(${todo:#$a}); echo "Actor_$a extracted"
    fi
  done
  sleep 20
done
echo ALLDONE
