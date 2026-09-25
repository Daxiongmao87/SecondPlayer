#!/bin/bash
# Chunked vision-quiz driver: fresh process per 8 questions (UR leak),
# one retry per chunk with jsonl truncation. Never stops the loop.
cd ~/projects/secondplayer && source xpu-env.sh
LOG=probes/2026-09-25-vision/vision-bf16.log
JSONL=probes/2026-09-25-vision/vision-bf16.jsonl
rm -f "$LOG" "$JSONL"
for OFF in 0 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136; do
  for TRY in 1 2; do
    NL=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
    echo "--- chunk off=$OFF try=$TRY (jsonl=$NL)" >> "$LOG"
    QOFFSET=$OFF QLIMIT=8 .venv/bin/python probes/2026-09-25-vision/quiz.py >> "$LOG" 2>&1 && break
    head -n "$NL" "$JSONL" > /tmp/vq.tmp 2>/dev/null && mv /tmp/vq.tmp "$JSONL" || rm -f "$JSONL"
    echo "chunk off=$OFF try=$TRY FAILED; truncated to $NL" >> "$LOG"
    sleep 5
  done
done
echo ALLDONE >> "$LOG"
