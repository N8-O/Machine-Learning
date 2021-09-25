#!/bin/bash
# Clone github repository
echo "Cloning Nathan's github Machine-Learning repository..."
git clone https://github.com/N8-O/Machine-Learning

cd Machine-Learning/DecisionTree

echo "Running Car Data Set"
python3.9 DT_car.py
echo "Running Bank Data Set with unknown as attribute"
python3.9 DT_bank_unknown_as_attribute.py
echo "Running Bank Data Set with unknown replaced"
python3.9 DT_bank_unknown_filled_common.py
