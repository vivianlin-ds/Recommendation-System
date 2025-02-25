import csv
import math
import sys

def read_predictions(file_path):
    predictions = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if present
        for row in reader:
            predictions.append(float(row[2]))
    return predictions


def calculate_rmse(predictions1, predictions2):
    if len(predictions1) != len(predictions2):
        raise ValueError("Both prediction lists must have the same length.")

    squared_errors = [(pred1 - pred2)**2 for pred1, pred2 in zip(predictions1, predictions2)]
    mean_squared_error = sum(squared_errors) / len(predictions1)
    rmse = math.sqrt(mean_squared_error)
    return rmse


def calculate_error_distribution(predictions, ground_truth):
    error_distribution = {'>=0 and <1': 0, '>=1 and <2': 0, '>=2 and <3': 0, '>=3 and <4': 0, '>=4': 0}

    for pred, truth in zip(predictions, ground_truth):
        diff = abs(pred - truth)
        if diff < 1:
            error_distribution['>=0 and <1'] += 1
        elif 1 <= diff < 2:
            error_distribution['>=1 and <2'] += 1
        elif 2 <= diff < 3:
            error_distribution['>=2 and <3'] += 1
        elif 3 <= diff < 4:
            error_distribution['>=3 and <4'] += 1
        else:
            error_distribution['>=4'] += 1

    return error_distribution


# Replace 'file1.csv' and 'file2.csv' with your actual file paths
file1_path = 'data/yelp_val.csv'
file2_path = sys.argv[1]

predictions1 = read_predictions(file1_path)
predictions2 = read_predictions(file2_path)

rmse = calculate_rmse(predictions1, predictions2)

print(f"RMSE between the two files: {rmse}")

error_distribution = calculate_error_distribution(predictions2, predictions1)

print("Error Distribution:")
for key, value in error_distribution.items():
    print(f"{key}: {value}")
