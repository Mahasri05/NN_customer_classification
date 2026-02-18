# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="688" height="1053" alt="image" src="https://github.com/user-attachments/assets/805538ad-4c68-4ba7-b8b0-1d13cc2cb57c" />

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
            nn.Linear(input_size, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.net(x)
        

```
```python
input_size = X_train.shape[1]

model = PeopleClassifier(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
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

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}"))

```



## Dataset Information

<img width="897" height="692" alt="image" src="https://github.com/user-attachments/assets/f8277156-c067-4332-94d0-152602c7ccb1" />

## OUTPUT



### Confusion Matrix

<img width="762" height="626" alt="image" src="https://github.com/user-attachments/assets/15892723-3287-413c-8793-24a3ef2e1b26" />

### Classification Report

<img width="642" height="327" alt="image" src="https://github.com/user-attachments/assets/8bacd60f-0484-4365-906c-4c12b73ed016" />


### New Sample Data Prediction

<img width="677" height="342" alt="image" src="https://github.com/user-attachments/assets/ac3bf92d-12e3-42cc-bfe4-cb05f68076d6" />

## RESULT
Thus the neural network classification model was successfully developed.

