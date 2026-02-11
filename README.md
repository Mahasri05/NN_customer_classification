# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1692" height="987" alt="image" src="https://github.com/user-attachments/assets/f69b02ca-877a-4cdd-900c-6c660380539c" />

## DESIGN STEPS

## STEP 1: Load Dataset

Load CSV file using pandas.

## STEP 2: Preprocess Data

Handle missing values

Encode categorical features

Normalize numerical features

Convert labels (A,B,C,D → 0,1,2,3)

## STEP 3: Split Dataset

Train-test split (80% train, 20% test)

## STEP 4: Create Neural Network

Define layers using PyTorch.

## STEP 5: Train Model

Use CrossEntropyLoss and Adam optimizer.

## STEP 6: Evaluate Model

Confusion matrix and classification report.

## STEP 7: Predict New Samples

Test with unseen data.


## PROGRAM

### Name: MAHASRI D
### Register Number: 212224220058

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)



    def forward(self, x):
        #Include your code here
        

```
```python
input_size = X_train.shape[1]

model = PeopleClassifier(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):

    for epoch in range(epochs):
        total_loss = 0

        for xb, yb in train_loader:

            optimizer.zero_grad()

            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

```



## Dataset Information

<img width="897" height="692" alt="image" src="https://github.com/user-attachments/assets/f8277156-c067-4332-94d0-152602c7ccb1" />

## OUTPUT



### Confusion Matrix

<img width="563" height="513" alt="image" src="https://github.com/user-attachments/assets/cb74a079-156c-4d2c-ad41-da1f278465f3" />

### Classification Report

<img width="595" height="257" alt="image" src="https://github.com/user-attachments/assets/12eba282-74b9-43d2-95e4-8be614f2f9b5" />


### New Sample Data Prediction

<img width="653" height="233" alt="image" src="https://github.com/user-attachments/assets/757ab009-b1a6-49aa-afd1-8c81b6cb3530" />

## RESULT
Thus the neural network classification model was successfully developed.

