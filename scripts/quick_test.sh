#!/bin/bash

echo "Running quick training test..."
python scripts/run_trainer.py \
    --dataset dummy \
    --num-epochs 3 \
    --batch-size 2 \
    --num-workers 0 \
    --log-interval 5 \
    --exp-name quick_test

echo ""
echo "Training complete! Check logs/quick_test_*