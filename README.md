# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/a61ed8d1-3c5b-46a5-9fae-4235d563dd6a)


## DESIGN STEPS

### STEP 1:
Understand the classification task and identify input and output variables.

### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.

### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.

### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

### STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).

### STEP 4:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

### STEP 5:
Save the trained model, export it if needed, and deploy it for real-world use.



## PROGRAM

### Name: SANDHIYA SREE B
### Register Number:212223220093

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self,x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
        

```
```python

# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)


```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):

      model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      output = model(X_batch)
      loss = criterion(output,y_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```



## Dataset Information

![Screenshot 2025-03-21 213628](https://github.com/user-attachments/assets/054433aa-b2e7-460d-8687-e9b91c09a608)


## OUTPUT
### Confusion Matrix

![Screenshot 2025-03-21 213720](https://github.com/user-attachments/assets/faf49c03-3081-467d-965e-9c2e68537557)

### Classification Report

![Screenshot 2025-03-21 213704](https://github.com/user-attachments/assets/f03c8c92-0b7b-43f1-abeb-a22d79b64864)



### New Sample Data Prediction

![Screenshot 2025-03-21 213735](https://github.com/user-attachments/assets/b4447e27-a9e8-465f-a20d-ab38fefa732a)


## RESULT
Thus a neural network classification model for the given dataset is executed successfully.
