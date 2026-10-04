# Artificial Neural Network Regression Project

## Prologue

This project is a practical example of using an Artificial Neural Network (ANN) for a regression problem. The notebook file, `ANN_Regression.ipynb`, demonstrates how a deep learning model can be trained to estimate a continuous target value from several input features. Instead of classifying a sample into one of several categories, this task predicts a numeric outcome — a classic regression problem.

The dataset used here is a power plant dataset containing several environmental variables. The goal is to learn the relationship between those variables and the target output known as the energy produced. The network does not simply memorize the numbers; it learns patterns and correlations between the inputs and the output, then generalizes to unseen data.

This project is a perfect beginner-friendly introduction to:

- supervised learning
- regression modeling
- feature selection
- data preprocessing
- train-test splitting
- standardization
- PyTorch model creation
- neural network training
- loss function optimization
- model evaluation and interpretation

The notebook is intentionally simple, compact, and educational. It shows the raw flow of a deep learning project from raw data to trained model without hiding the important fundamentals.

---

## Chapter 1: Understanding the Problem

The task in this notebook is not classification but regression. In classification, the model predicts discrete labels such as "cat" vs "dog" or "spam" vs "not spam." In regression, the model predicts a continuous numerical value.

Here, the value being predicted is called `PE`, which stands for the produced energy. The input variables include:

- `AT`: ambient temperature
- `V`: vacuum
- `AP`: ambient pressure
- `RH`: relative humidity
- `PE`: produced energy (target)

The relationship among these variables is not random. For example, temperature, pressure, and humidity all affect the power generation process. The job of the ANN is to infer this numerical relationship automatically from historical data.

A regression problem can be formally described as:

$$
\hat{y} = f(x_1, x_2, x_3, x_4)
$$

where:

- $x_1, x_2, x_3, x_4$ are the feature inputs
- $f$ is the learned function represented by the neural network
- $\hat{y}$ is the predicted output value

The network attempts to minimize the difference between predicted values and actual target values.

---

## Chapter 2: Loading the Dataset

The first step in the notebook is to import the necessary libraries:

```python
import pandas as pd
import numpy as np
```

`pandas` helps read and manipulate tabular data. `numpy` supports numerical operations and array handling. These are the backbone of data science workflows.

Then the dataset is loaded using:

```python
df = pd.read_csv("powerplant_data.csv")
```

This is a CSV file containing all observations. CSV stands for Comma-Separated Values, and it stores tabular data in a simple text format. Once loaded into a DataFrame, the dataset becomes easy to explore, filter, and model.

The notebook then displays the first few rows using:

```python
df.head()
```

This is a crucial step because it allows us to confirm that the dataset has the expected structure and that columns are named correctly before training a model.

---

## Chapter 3: Data Understanding and Column Meaning

The dataset is a simplified power plant dataset, and each row represents one observation from the plant. The variables have the following meaning:

- `AT` (Ambient Temperature): environmental temperature, often in degrees Celsius
- `V` (Vacuum): pressure or vacuum level in the turbine system
- `AP` (Ambient Pressure): atmospheric pressure
- `RH` (Relative Humidity): water vapor content in the air
- `PE` (Power Output / Produced Energy): the target variable to be predicted

The comment block in the notebook explains this clearly:

```python
# AT=> temperature
# v=> vacuum
# AP=> pressure
# RH=> humidity
# PE=> produce energy
```

This gives a conceptual map of what each feature corresponds to in the real world. A regression model learns how these conditions influence the power output.

---

## Chapter 4: Checking Missing Values

The notebook checks for missing values using:

```python
df.isnull().sum()
```

This is a standard data-quality step. Missing values are common in real datasets and can cause errors or biased models if not handled properly. If any column had missing values, they would need to be filled or removed before training.

This notebook confirms that the data is clean enough for modeling, which is ideal for a beginner example.

---

## Chapter 5: Features and Target Separation

The next step separates the dataset into independent variables and dependent variables:

```python
X = df.drop("PE", axis=1)
y = df["PE"]
```

Here:

- `X` contains all features except the target
- `y` contains the output value we want to predict

This is a standard supervised learning setup. The model will use the features in `X` to learn how to predict `y`.

By dropping `PE`, the model is prevented from seeing the target directly and is forced to infer it from the remaining columns.

---

## Chapter 6: Train-Test Split

The notebook then splits the data into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

This is one of the most important steps in machine learning.

Why split the dataset?

- If the model is trained and tested on the same data, it may appear highly accurate but fail on unseen data.
- A train-test split gives a realistic estimate of how the model performs in the real world.

Here:

- `80%` of the data is used for training
- `20%` is held out for testing
- `random_state=42` ensures reproducibility; the same split is produced each time

This helps to reduce the risk of random data ordering affecting results.

---

## Chapter 7: Data Shape and Dimensions

The notebook also checks:

```python
df.shape
```

This returns the number of rows and columns in the dataset. For example, if the shape is `(9568, 5)`, it means there are 9568 rows and 5 columns. This gives an understanding of dataset size before building the model.

Knowledge of the data size is important because it influences:

- how much time training may take
- whether the dataset is large enough for deep learning
- whether the network should be simple or complex

This dataset is modest in size, which makes it ideal for a beginner-friendly neural network model.

---

## Chapter 8: Standardization with StandardScaler

Before training the neural network, the notebook normalizes the input features:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This is a very important preprocessing step.

### Why scaling matters

Neural networks are sensitive to the magnitude of input values. If one feature is in the range of 10s and another is in the range of 1000s, the model may learn unstable or biased patterns. Standardization fixes this by converting each feature to a distribution with:

- mean = 0
- standard deviation = 1

The transformation formula is:

$$
z = \frac{x - \mu}{\sigma}
$$

where:

- $x$ is the original value
- $\mu$ is the mean of the feature
- $\sigma$ is the standard deviation

This makes optimization easier because each feature contributes more evenly during gradient descent.

### Why fit on training data only?

The scaler is fit on the training set and then applied to the test set. This prevents information leakage from the test set into the model training process. In real machine learning pipelines, learn preprocessing parameters only from the training data.

---

## Chapter 9: Converting to PyTorch Tensors

The next stage converts the scaled arrays into PyTorch tensors:

```python
import torch
import torch.nn as nn

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
```

This is where PyTorch enters the picture.

### Why tensors?

Tensors are the fundamental data structures in PyTorch. They are like NumPy arrays but with support for automatic differentiation, which is essential for training neural networks.

### Why reshape target values to `(-1, 1)`?

The target `y_train` is a one-dimensional array. A neural network output layer often produces a matrix of shape `(batch_size, 1)`. Reshaping ensures compatibility with the loss function and the architecture.

This shape is important because the model will output a single value for each sample, representing the predicted energy output.

---

## Chapter 10: Building DataLoaders

The notebook wraps the tensors in a `TensorDataset` and `DataLoader`:

```python
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
```

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
```

### What is a DataLoader?

A DataLoader batches data into mini-batches and shuffles training samples for stochastic optimization. This is important because neural networks are trained more efficiently with batch-wise updates rather than processing all data at once.

### Why use batch size 32?

A batch size of 32 is a common default choice:

- it is small enough for efficient computation
- it provides stable gradient updates
- it is large enough to reduce noisy updates

Shuffling training data helps the model avoid learning from a rigid ordering of samples.

---

## Chapter 11: Introducing the Neural Network

The network is defined as a custom class inheriting from `nn.Module`:

```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(X_train.shape[1], 6),
            nn.ReLU(),

            nn.Linear(6, 6),
            nn.ReLU(),

            nn.Linear(6, 1)
        )

    def forward(self, x):
        return self.model(x)
```

This is a small feedforward neural network, also called a Multi-Layer Perceptron (MLP).

### Structure of the network

1. Input layer
   - The number of input nodes equals the number of features.
   - Here, there are 4 input features: `AT`, `V`, `AP`, and `RH`.

2. Hidden layer 1
   - `nn.Linear(X_train.shape[1], 6)`
   - Maps 4 inputs to 6 neurons
   - `ReLU` is applied after this layer

3. Hidden layer 2
   - `nn.Linear(6, 6)`
   - Another transformation with 6 neurons
   - `ReLU` again introduces non-linearity

4. Output layer
   - `nn.Linear(6, 1)`
   - Produces one prediction for each sample

### Why ReLU?

`ReLU` stands for Rectified Linear Unit and is defined as:

$$
\text{ReLU}(x) = \max(0, x)
$$

It is popular because it is simple, efficient, and helps the network learn non-linear relationships. Without non-linearity, the network would behave like a linear model regardless of depth.

This network can learn a function that approximates the nonlinear relationship between environmental conditions and power output.

---

## Chapter 12: The Loss Function and Optimizer

The model is compiled conceptually with a loss function and optimizer:

```python
import torch.optim as optim

model = ANN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
```

### Mean Squared Error (MSE)

`nn.MSELoss()` computes the mean squared error:

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

This measures the average squared difference between predicted and actual values. Squaring the error emphasizes larger errors, which is useful in regression tasks.

### Adam optimizer

`Adam` is an adaptive optimization algorithm used to update the network weights based on the gradients computed by backpropagation. It keeps the training stable and efficient even when features vary in scale.

---

## Chapter 13: The Training Loop

The notebook trains the model for 100 epochs:

```python
train_losses = []
val_losses = []
best_val_loss = float("inf")
epochs = 100

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(test_loader)
    val_losses.append(epoch_val_loss)
    print(f"epoch {epoch+1}/{epochs} ==> training loss = {epoch_train_loss} & val loss = {epoch_val_loss}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_model.pt")
```

This is the heart of deep learning.

### Step-by-step breakdown

#### 1. Set model to training mode

```python
model.train()
```

This ensures layers such as dropout or batch normalization behave in training mode. Here, since simple layers are used, this is mostly standard practice.

#### 2. Forward pass

```python
outputs = model(xb)
```

The inputs pass through the hidden layers and final output layer to produce predictions.

#### 3. Compute loss

```python
loss = criterion(outputs, yb)
```

This quantifies how far the predictions are from the true values.

#### 4. Backpropagation

```python
loss.backward()
```

The model computes gradients of the loss with respect to all weights. This tells the model how to adjust its parameters to reduce error.

#### 5. Gradient descent update

```python
optimizer.step()
```

The optimizer updates model weights using the calculated gradients.

#### 6. Validation pass

The notebook evaluates the model on the test set after each training epoch. This helps monitor whether the model is improving or overfitting.

### Why save the best model?

The code stores the best model weights when validation loss reaches a new minimum:

```python
torch.save(model.state_dict(), "best_model.pt")
```

This protects the model from later training epochs that may perform worse. It is a common practice in deep learning to preserve the best-performing checkpoint.

---

## Chapter 14: Visualizing the Training Process

The notebook then plots the training and validation loss curves:

```python
import matplotlib.pyplot as plt

loss_df = pd.DataFrame({
    "Training Loss": train_losses,
    "Validation Loss": val_losses
})

plt.figure(figsize=(12, 8))
plt.plot(loss_df["Training Loss"], label="Training Loss")
plt.plot(loss_df["Validation Loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()
```

This graph is especially important because it shows the learning behavior over time.

### Interpreting the curve

- If training loss decreases steadily, the model is learning.
- If validation loss decreases too, the model is generalizing well.
- If training loss keeps dropping while validation loss begins increasing, the model may be overfitting.

The graph helps the practitioner understand the training dynamics at a glance.

---

## Chapter 15: Loading the Best Model

After training, the notebook loads the stored best weights:

```python
model.load_state_dict(torch.load("best_model.pt"))
```

This is important because the last epoch may not be the best epoch. By restoring the best checkpoint, the final model is the one with the lowest validation loss.

---

## Chapter 16: Evaluating Model Performance

The evaluation step is:

```python
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)
    train_mse_loss = criterion(train_preds, y_train_tensor)
    test_mse_loss = criterion(test_preds, y_test_tensor)

print("Training MSE:", train_mse_loss.item())
print("Testing Mse:", test_mse_loss.item())
```

### Why `torch.no_grad()`?

The `no_grad()` context disables gradient tracking. This is essential during evaluation because we do not want PyTorch to build unnecessary computational graphs when the model is only being tested.

### Why MSE?

MSE gives a numerical sense of prediction error. Lower values indicate better performance. However, it is useful to also look at coefficient of determination (`R²`) because MSE alone can be harder to interpret in context.

---

## Chapter 17: R-Squared Score

The notebook calculates the $R^2$ score:

```python
from sklearn.metrics import r2_score

print("r2 score =", r2_score(y_test, test_preds))
```

The coefficient of determination $R^2$ is defined as:

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

where:

- $y_i$ is the actual value
- $\hat{y}_i$ is the predicted value
- $\bar{y}$ is the mean of actual values

### Interpretation of $R^2$

- $R^2 = 1$: perfect prediction
- $R^2 = 0$: model predicts the mean value only
- negative $R^2$: model is worse than a simple baseline

This metric is often more intuitive than MSE because it tells us how much of the variation in the target is explained by the model.

---

## Chapter 18: Comparing Predictions to Actual Values

The notebook creates a DataFrame of actual versus predicted values:

```python
predicted_df = pd.DataFrame(test_preds.numpy(), columns=["Predicted Values"])
actual_df = pd.DataFrame(y_test.values, columns=["Actual Values"])
pd.concat([predicted_df, actual_df], axis=1)
```

This gives a side-by-side view of the expected and actual outputs. These tables are often used for real-world interpretation because a model may have an acceptable overall MSE while still making large mistakes on some samples.

This kind of comparison is a good final sanity check before concluding that the model is good enough for deployment or further tuning.

---

## Chapter 19: Why This Project Matters

This notebook is valuable because it teaches the complete lifecycle of a neural network regression model in a simple, readable way.

It covers the following essential pipeline:

1. Loading and inspecting data
2. Understanding features and target variable
3. Checking missing data
4. Splitting data into train and test sets
5. Standardizing features
6. Converting data to PyTorch tensors
7. Building a neural network architecture
8. Defining a loss function and optimizer
9. Training for many epochs
10. Saving the best model
11. Evaluating the model using MSE and R²
12. Interpreting the results visually and numerically

These are the same core principles used in far more complex deep learning systems in industry and research.

---

## Chapter 20: Conceptual Summary

At the highest level, this project demonstrates a fundamental truth in machine learning:

A neural network is simply a function approximator.

It learns by adjusting internal weights using gradient descent so that its predictions increasingly match the true target values. The model is not magic; it is a sequence of matrix multiplications, nonlinear activations, and optimization steps driven by data.

The notebook is a simplified but meaningful example of how computers can learn patterns from data and generalize to new values. In this case, the patterns are the relationships between environmental factors and the energy produced by a power plant.

---

## Final Reflection

This project is a strong starting point for understanding deep learning in practice. It shows how code, mathematics, and intuition come together in a real regression problem. Even though the network is small and the dataset is simple, the ideas behind it are the same as those behind large-scale systems such as image recognition, time-series forecasting, language modeling, and autonomous decision-making systems.

The essential lesson is this:

> Neural networks learn from data, improve through optimization, and are evaluated by how well they generalize to unseen examples.

That is the central idea behind the entire field of deep learning.

---

## Appendix: Core Python Concepts Used in the Notebook

- `pd.read_csv()` for loading tabular data
- `df.head()` for quick inspection
- `df.isnull().sum()` for missing-value checks
- `train_test_split()` for data partitioning
- `StandardScaler()` for feature normalization
- `TensorDataset` and `DataLoader` for batched training
- `nn.Sequential()` for building a neural network
- `nn.Linear()` for fully connected layers
- `nn.ReLU()` for non-linearity
- `nn.MSELoss()` for regression loss
- `optim.Adam()` for optimization
- `loss.backward()` for gradient computation
- `optimizer.step()` for updating weights
- `model.eval()` for evaluation mode
- `torch.save()` and `torch.load()` for checkpointing
- `r2_score()` for performance evaluation

These tools form the standard workflow of a PyTorch-based machine learning project.

---

## Closing Note

This README is intended to act as a guided companion to the notebook. If you read the notebook cell by cell alongside this explanation, the full learning experience becomes much clearer. The notebook is small, but the ideas it contains are foundational and remain relevant across the entire field of artificial intelligence and machine learning.

The future of AI is built on understanding such pipelines — from data to model to evaluation — and this project is a clear and approachable introduction to that process.
