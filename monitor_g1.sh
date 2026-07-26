#!/bin/bash
# G1 recording monitor — 每 50 episodes 报告一次进度 + 剩余时间估算
TRACE_DIR="experiments/e1/traces/bf16/tau_bench"
TARGET=1320

BASE=$(ls "$TRACE_DIR"/*.json 2>/dev/null | wc -l)
BATCH_START_COUNT=$BASE
BATCH_START_TIME=$(date +%s)
LAST_MILESTONE=$(( (BASE / 50) * 50 ))

echo "============================================================"
echo "  G1 Monitor started at $(date '+%H:%M:%S')"
echo "  Baseline: $BASE / $TARGET episodes already recorded"
echo "============================================================"
echo ""

while true; do
    sleep 30
    N=$(ls "$TRACE_DIR"/*.json 2>/dev/null | wc -l)

    MILESTONE=$(( (N / 50) * 50 ))
    if [ "$MILESTONE" -gt "$LAST_MILESTONE" ] && [ "$N" -ge "$MILESTONE" ]; then
        NOW=$(date +%s)
        BATCH_ELAPSED=$(( NOW - BATCH_START_TIME ))
        BATCH_MIN=$(( BATCH_ELAPSED / 60 ))
        BATCH_SEC=$(( BATCH_ELAPSED % 60 ))
        BATCH_EPS=$(( MILESTONE - LAST_MILESTONE ))

        # Rate calculation for THIS batch
        if [ $BATCH_ELAPSED -gt 0 ]; then
            RATE=$(python3 -c "print(f'{3600 * $BATCH_EPS / $BATCH_ELAPSED:.1f}')")
        else
            RATE="N/A"
        fi

        # Overall rate since start
        TOTAL_ELAPSED=$(( NOW - BATCH_START_TIME + 0 ))  # will fix below
        # Actually use overall from script start
        SCRIPT_ELAPSED=$(( (NOW - BATCH_START_TIME) / 60 ))

        # ETA based on batch rate
        REMAINING=$(( TARGET - N ))
        if [ "$RATE" != "N/A" ]; then
            ETA_HOURS=$(python3 -c "print(f'{int($REMAINING / float($RATE))}h {int(60 * ($REMAINING / float($RATE) - int($REMAINING / float($RATE))))}m')")
        else
            ETA_HOURS="N/A"
        fi

        PCT=$(python3 -c "print(f'{100*$N/$TARGET:.1f}')")

        echo ">>> MILESTONE ${MILESTONE}/${TARGET} (${PCT}%) $(date '+%H:%M:%S')"
        echo "    +${BATCH_EPS} episodes in ${BATCH_MIN}m${BATCH_SEC}s | rate: ${RATE} eps/h"
        echo "    ETA: ${ETA_HOURS} remaining"
        echo ""

        LAST_MILESTONE=$MILESTONE
        BATCH_START_COUNT=$MILESTONE
        BATCH_START_TIME=$NOW
    fi

    # Every 30 min also print a heartbeat
    NOW=$(date +%s)
    HEARTBEAT_INTERVAL=1800
    if [ $(( NOW - BATCH_START_TIME )) -ge $HEARTBEAT_INTERVAL ]; then
        echo "--- heartbeat $(date '+%H:%M:%S') | $N/$TARGET | ${PCT}% ---"
        BATCH_START_TIME=$NOW
    fi
done
