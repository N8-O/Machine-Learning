#! /bin/bash
echo "Running Practice 2a"
python3.9 AdaBoost.py
echo "Running Practice 2b"
python3.9 run_BaggedTreesB.py
echo "Running Practice 2c"
python3.9 BaggedTreeVarianceandBias.py
echo "Running Practice 2d"
python3.9 run_RandForest.py
echo "Running Practice 2e"
python3.9 RandomTreeVarianceandBias.py
cd ..
cd Linear\ \ Regression/
echo "Running Practice 4"
python3.9 grad_descent_algs.py
