import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

print("TensorFlow version:", tf.__version__)

# ============================================================
# LOAD IMDB DATASET
# ============================================================

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(
    num_words=10000
)

print("Training samples:", len(train_data))
print("Test samples:", len(test_data))


# ============================================================
# VECTORIZE THE DATA
# ============================================================

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0

    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)


# ============================================================
# CREATE TRAINING AND VALIDATION SETS
# ============================================================

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("Training samples:", len(partial_x_train))
print("Validation samples:", len(x_val))
print("Test samples:", len(x_test))


# ============================================================
# FUNCTION TO CREATE A MODEL
# ============================================================

def create_model(hidden_units):
    model = keras.Sequential([
        layers.Input(shape=(10000,)),
        layers.Dense(hidden_units[0], activation="relu"),
        layers.Dense(hidden_units[1], activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# BASELINE MODEL
# Two hidden layers, 16 units each
# ============================================================

print("\n" + "=" * 60)
print("BASELINE MODEL - TWO HIDDEN LAYERS")
print("=" * 60)

baseline_model = create_model([16, 16])

print(baseline_model.summary())

baseline_history = baseline_model.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

baseline_history_dict = baseline_history.history

baseline_best_val = max(baseline_history_dict["val_accuracy"])
baseline_best_epoch = (
    baseline_history_dict["val_accuracy"].index(baseline_best_val) + 1
)

baseline_test_loss, baseline_test_accuracy = baseline_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Baseline best validation accuracy:", baseline_best_val)
print("Baseline best epoch:", baseline_best_epoch)
print("Baseline test accuracy:", baseline_test_accuracy)


# ============================================================
# ONE HIDDEN LAYER
# ============================================================

print("\n" + "=" * 60)
print("ONE HIDDEN LAYER MODEL")
print("=" * 60)

one_hidden_model = keras.Sequential([
    layers.Input(shape=(10000,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

one_hidden_model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(one_hidden_model.summary())

one_hidden_history = one_hidden_model.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

one_hidden_history_dict = one_hidden_history.history

one_hidden_best_val = max(one_hidden_history_dict["val_accuracy"])
one_hidden_best_epoch = (
    one_hidden_history_dict["val_accuracy"].index(one_hidden_best_val) + 1
)

print("One hidden layer best validation accuracy:", one_hidden_best_val)
print("One hidden layer best epoch:", one_hidden_best_epoch)


# ============================================================
# THREE HIDDEN LAYERS
# ============================================================

print("\n" + "=" * 60)
print("THREE HIDDEN LAYER MODEL")
print("=" * 60)

three_hidden_model = keras.Sequential([
    layers.Input(shape=(10000,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

three_hidden_model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(three_hidden_model.summary())

three_hidden_history = three_hidden_model.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

three_hidden_history_dict = three_hidden_history.history

three_hidden_best_val = max(three_hidden_history_dict["val_accuracy"])
three_hidden_best_epoch = (
    three_hidden_history_dict["val_accuracy"].index(three_hidden_best_val) + 1
)

three_hidden_test_loss, three_hidden_test_accuracy = (
    three_hidden_model.evaluate(
        x_test,
        y_test,
        verbose=0
    )
)

print("Three hidden layer best validation accuracy:", three_hidden_best_val)
print("Three hidden layer best epoch:", three_hidden_best_epoch)
print("Three hidden layer test accuracy:", three_hidden_test_accuracy)


# ============================================================
# 32 UNIT MODEL
# Two hidden layers, 32 units each
# ============================================================

print("\n" + "=" * 60)
print("32 UNIT MODEL")
print("=" * 60)

model_32 = create_model([32, 32])

print(model_32.summary())

history_32 = model_32.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

history_32_dict = history_32.history

best_32_val = max(history_32_dict["val_accuracy"])
best_32_epoch = (
    history_32_dict["val_accuracy"].index(best_32_val) + 1
)

test_32_loss, test_32_accuracy = model_32.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("32-unit best validation accuracy:", best_32_val)
print("32-unit best epoch:", best_32_epoch)
print("32-unit test accuracy:", test_32_accuracy)


# ============================================================
# 64 UNIT MODEL
# Two hidden layers, 64 units each
# ============================================================

print("\n" + "=" * 60)
print("64 UNIT MODEL")
print("=" * 60)

model_64 = create_model([64, 64])

print(model_64.summary())

history_64 = model_64.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

history_64_dict = history_64.history

best_64_val = max(history_64_dict["val_accuracy"])
best_64_epoch = (
    history_64_dict["val_accuracy"].index(best_64_val) + 1
)

test_64_loss, test_64_accuracy = model_64.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("64-unit best validation accuracy:", best_64_val)
print("64-unit best epoch:", best_64_epoch)
print("64-unit test accuracy:", test_64_accuracy)


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame({
    "Model": [
        "One Hidden Layer",
        "Baseline - Two Hidden Layers",
        "Three Hidden Layers",
        "32 Units",
        "64 Units"
    ],

    "Hidden Layers": [
        1,
        2,
        3,
        2,
        2
    ],

    "Hidden Units": [
        "16",
        "16, 16",
        "16, 16, 16",
        "32, 32",
        "64, 64"
    ],

    "Best Validation Accuracy": [
        one_hidden_best_val,
        baseline_best_val,
        three_hidden_best_val,
        best_32_val,
        best_64_val
    ],

    "Best Epoch": [
        one_hidden_best_epoch,
        baseline_best_epoch,
        three_hidden_best_epoch,
        best_32_epoch,
        best_64_epoch
    ],

    "Test Accuracy": [
        np.nan,
        baseline_test_accuracy,
        three_hidden_test_accuracy,
        test_32_accuracy,
        test_64_accuracy
    ]
})

results["Best Validation Accuracy"] = (
    results["Best Validation Accuracy"] * 100
)

results["Test Accuracy"] = (
    results["Test Accuracy"] * 100
)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(results.to_string(index=False))


# ============================================================
# GRAPH - VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["Best Validation Accuracy"]
)

plt.ylabel("Best Validation Accuracy (%)")
plt.xlabel("Model")
plt.title("IMDB Neural Network Validation Accuracy")
plt.xticks(rotation=30, ha="right")
plt.ylim(80, 95)

plt.tight_layout()
plt.show()

# ============================================================
# MSE LOSS MODEL
# Same architecture as baseline, but MSE loss
# ============================================================

print("\n" + "=" * 60)
print("MSE LOSS MODEL")
print("=" * 60)

mse_model = keras.Sequential([
    layers.Input(shape=(10000,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

mse_model.compile(
    optimizer="rmsprop",
    loss="mse",
    metrics=["accuracy"]
)

print(mse_model.summary())

mse_history = mse_model.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

mse_history_dict = mse_history.history

best_mse_val = max(mse_history_dict["val_accuracy"])
best_mse_epoch = (
    mse_history_dict["val_accuracy"].index(best_mse_val) + 1
)

mse_test_loss, mse_test_accuracy = mse_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("MSE best validation accuracy:", best_mse_val)
print("MSE best epoch:", best_mse_epoch)
print("MSE test accuracy:", mse_test_accuracy)

# ============================================================
# TANH ACTIVATION MODEL
# Same architecture as baseline, but tanh activation
# ============================================================

print("\n" + "=" * 60)
print("TANH ACTIVATION MODEL")
print("=" * 60)

tanh_model = keras.Sequential([
    layers.Input(shape=(10000,)),
    layers.Dense(16, activation="tanh"),
    layers.Dense(16, activation="tanh"),
    layers.Dense(1, activation="sigmoid")
])

tanh_model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(tanh_model.summary())

tanh_history = tanh_model.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

tanh_history_dict = tanh_history.history

best_tanh_val = max(tanh_history_dict["val_accuracy"])
best_tanh_epoch = (
    tanh_history_dict["val_accuracy"].index(best_tanh_val) + 1
)

tanh_test_loss, tanh_test_accuracy = tanh_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Tanh best validation accuracy:", best_tanh_val)
print("Tanh best epoch:", best_tanh_epoch)
print("Tanh test accuracy:", tanh_test_accuracy)

# ============================================================
# TANH ACTIVATION MODEL
# ============================================================

print("\n" + "=" * 60)
print("TANH ACTIVATION MODEL")
print("=" * 60)

tanh_model = keras.Sequential([
    layers.Input(shape=(10000,)),
    layers.Dense(16, activation="tanh"),
    layers.Dense(16, activation="tanh"),
    layers.Dense(1, activation="sigmoid")
])

tanh_model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

tanh_history = tanh_model.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

best_tanh_val = max(tanh_history.history["val_accuracy"])
best_tanh_epoch = (
    tanh_history.history["val_accuracy"].index(best_tanh_val) + 1
)

tanh_test_loss, tanh_test_accuracy = tanh_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Tanh best validation accuracy:", best_tanh_val)
print("Tanh best epoch:", best_tanh_epoch)
print("Tanh test accuracy:", tanh_test_accuracy)


# ============================================================
# DROPOUT 0.2
# ============================================================

print("\n" + "=" * 60)
print("DROPOUT 0.2 MODEL")
print("=" * 60)

dropout20_model = keras.Sequential([
    layers.Input(shape=(10000,)),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

dropout20_model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

dropout20_history = dropout20_model.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

best_dropout20_val = max(
    dropout20_history.history["val_accuracy"]
)

best_dropout20_epoch = (
    dropout20_history.history["val_accuracy"].index(
        best_dropout20_val
    ) + 1
)

dropout20_test_loss, dropout20_test_accuracy = (
    dropout20_model.evaluate(
        x_test,
        y_test,
        verbose=0
    )
)

print("Dropout 0.2 best validation accuracy:", best_dropout20_val)
print("Dropout 0.2 best epoch:", best_dropout20_epoch)
print("Dropout 0.2 test accuracy:", dropout20_test_accuracy)


# ============================================================
# DROPOUT 0.5
# ============================================================

print("\n" + "=" * 60)
print("DROPOUT 0.5 MODEL")
print("=" * 60)

dropout50_model = keras.Sequential([
    layers.Input(shape=(10000,)),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

dropout50_model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

dropout50_history = dropout50_model.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

best_dropout50_val = max(
    dropout50_history.history["val_accuracy"]
)

best_dropout50_epoch = (
    dropout50_history.history["val_accuracy"].index(
        best_dropout50_val
    ) + 1
)

dropout50_test_loss, dropout50_test_accuracy = (
    dropout50_model.evaluate(
        x_test,
        y_test,
        verbose=0
    )
)

print("Dropout 0.5 best validation accuracy:", best_dropout50_val)
print("Dropout 0.5 best epoch:", best_dropout50_epoch)
print("Dropout 0.5 test accuracy:", dropout50_test_accuracy)


# ============================================================
# L2 REGULARIZATION
# ============================================================

print("\n" + "=" * 60)
print("L2 REGULARIZATION MODEL")
print("=" * 60)

l2_model = keras.Sequential([
    layers.Input(shape=(10000,)),
    layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001)
    ),
    layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001)
    ),
    layers.Dense(1, activation="sigmoid")
])

l2_model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

l2_history = l2_model.fit(
    partial_x_train,
    partial_y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

best_l2_val = max(
    l2_history.history["val_accuracy"]
)

best_l2_epoch = (
    l2_history.history["val_accuracy"].index(
        best_l2_val
    ) + 1
)

l2_test_loss, l2_test_accuracy = l2_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("L2 best validation accuracy:", best_l2_val)
print("L2 best epoch:", best_l2_epoch)
print("L2 test accuracy:", l2_test_accuracy)


# ============================================================
# COMPLETE RESULTS TABLE
# ============================================================

results = pd.DataFrame({
    "Model": [
        "One Hidden Layer",
        "Baseline - 2 Layers",
        "Three Hidden Layers",
        "32 Units",
        "64 Units",
        "MSE Loss",
        "Tanh Activation",
        "Dropout 0.2",
        "Dropout 0.5",
        "L2 Regularization"
    ],

    "Hidden Layers": [
        1, 2, 3, 2, 2, 2, 2, 2, 2, 2
    ],

    "Hidden Units": [
        "16",
        "16, 16",
        "16, 16, 16",
        "32, 32",
        "64, 64",
        "16, 16",
        "16, 16",
        "16, 16",
        "16, 16",
        "16, 16"
    ],

    "Change": [
        "Architecture",
        "Baseline",
        "Architecture",
        "Units",
        "Units",
        "MSE Loss",
        "Tanh",
        "Dropout",
        "Dropout",
        "L2"
    ],

    "Best Validation Accuracy": [
        one_hidden_best_val,
        baseline_best_val,
        three_hidden_best_val,
        best_32_val,
        best_64_val,
        best_mse_val,
        best_tanh_val,
        best_dropout20_val,
        best_dropout50_val,
        best_l2_val
    ],

    "Best Epoch": [
        one_hidden_best_epoch,
        baseline_best_epoch,
        three_hidden_best_epoch,
        best_32_epoch,
        best_64_epoch,
        best_mse_epoch,
        best_tanh_epoch,
        best_dropout20_epoch,
        best_dropout50_epoch,
        best_l2_epoch
    ],

    "Test Accuracy": [
        np.nan,
        baseline_test_accuracy,
        three_hidden_test_accuracy,
        test_32_accuracy,
        test_64_accuracy,
        mse_test_accuracy,
        tanh_test_accuracy,
        dropout20_test_accuracy,
        dropout50_test_accuracy,
        l2_test_accuracy
    ]
})

results["Best Validation Accuracy"] *= 100
results["Test Accuracy"] *= 100

print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(results.to_string(index=False))


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

results.to_csv(
    "IMDB_Neural_Network_Results.csv",
    index=False
)

print("\nResults saved to:")
print("IMDB_Neural_Network_Results.csv")


# ============================================================
# VALIDATION ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(12, 7))

plt.bar(
    results["Model"],
    results["Best Validation Accuracy"]
)

plt.ylabel("Best Validation Accuracy (%)")
plt.xlabel("Model")
plt.title("IMDB Neural Network Model Comparison")

plt.xticks(
    rotation=40,
    ha="right"
)

plt.ylim(80, 95)

plt.tight_layout()

plt.savefig(
    "IMDB_Validation_Accuracy_Comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# TEST ACCURACY GRAPH
# ============================================================

test_results = results.dropna(
    subset=["Test Accuracy"]
)

plt.figure(figsize=(12, 7))

plt.bar(
    test_results["Model"],
    test_results["Test Accuracy"]
)

plt.ylabel("Test Accuracy (%)")
plt.xlabel("Model")
plt.title("IMDB Neural Network Test Accuracy")

plt.xticks(
    rotation=40,
    ha="right"
)

plt.ylim(80, 95)

plt.tight_layout()

plt.savefig(
    "IMDB_Test_Accuracy_Comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# FIND BEST VALIDATION MODEL
# ============================================================

best_model_row = results.loc[
    results["Best Validation Accuracy"].idxmax()
]

print("\n" + "=" * 60)
print("BEST VALIDATION MODEL")
print("=" * 60)

print(
    "Model:",
    best_model_row["Model"]
)

print(
    "Validation accuracy:",
    f'{best_model_row["Best Validation Accuracy"]:.2f}%'
)

print(
    "Best epoch:",
    int(best_model_row["Best Epoch"])
)

print("\nAssignment experiments complete.")