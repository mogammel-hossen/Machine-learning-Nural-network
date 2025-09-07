import csv
import math
import random

def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        next(lines)
        for row in lines:
            if row:  # skip empty rows
                # Take only columns 1-4 as features, and last column as class
                dataset.append(row[1:5] + [row[-1]])
    return dataset

# Convert features to float, keep last column (class label) as string
def preprocess_dataset(dataset):
    for i in range(len(dataset)):
        dataset[i][:4] = [float(x) for x in dataset[i][:4]]
    return dataset

# -------------------------------
# Split dataset
# -------------------------------
def split_dataset(dataset, train_ratio=0.7):
    dataset_copy = dataset[:]
    random.shuffle(dataset_copy)
    train_size = int(len(dataset) * train_ratio)
    train_set = dataset_copy[:train_size]
    test_set = dataset_copy[train_size:]
    return train_set, test_set

# -------------------------------
# Training (Calculate mean, stdev)
# -------------------------------
def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        vector = row[:-1]
        label = row[-1]
        if label not in separated:
            separated[label] = []
        separated[label].append(vector)
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize_dataset(dataset):
    summaries = [(mean(col), stdev(col)) for col in zip(*dataset)]
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

# -------------------------------
# Prediction
# -------------------------------
def calculate_probability(x, mean, stdev):
    if stdev == 0:  # avoid divide by zero
        return 1.0 if x == mean else 0.0
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row, priors):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = priors[class_value]
        for i in range(len(class_summaries)):
            mean_val, stdev_val = class_summaries[i]
            x = row[i]
            probabilities[class_value] *= calculate_probability(x, mean_val, stdev_val)
    # Normalize probabilities
    total = sum(probabilities.values())
    for c in probabilities:
        probabilities[c] /= total
    return probabilities

def predict(summaries, row, priors):
    probabilities = calculate_class_probabilities(summaries, row, priors)
    predicted_class = max(probabilities, key=probabilities.get)
    return predicted_class, probabilities

# -------------------------------
# Evaluate Model
# -------------------------------
def get_priors(train_set):
    class_counts = {}
    for row in train_set:
        label = row[-1]
        class_counts[label] = class_counts.get(label, 0) + 1
    total = len(train_set)
    return {label: count / total for label, count in class_counts.items()}

def evaluate(train_set, test_set):
    summaries = summarize_by_class(train_set)
    priors = get_priors(train_set)

    correct = 0
    for row in test_set:
        features = row[:-1]
        true_label = row[-1]
        predicted, probs = predict(summaries, features, priors)
        status = "Correct" if predicted == true_label else "Wrong"
        if status == "Correct":
            correct += 1

        print(f"Test Sample: {features}")
        print(f"Predicted: {predicted}")
        print(f"Probabilities: {probs}")
        print(status)
        print("-" * 40)

    accuracy = correct / len(test_set) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    filename = "Iris.csv"  # Make sure Iris.csv is in the same folder
    dataset = load_csv(filename)
    dataset = preprocess_dataset(dataset)

    # Set training ratio
    train_ratio = float(input("Enter training ratio (e.g. 0.7 for 70%): "))
    train_set, test_set = split_dataset(dataset, train_ratio)

    evaluate(train_set, test_set)
