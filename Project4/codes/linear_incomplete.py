## 2.8 1-layer linear model
# Hyper-parameters
input_size = 3*64*64
output_size = 10
num_epochs = 20
learning_rate = 0.001
batch_size = 200

# reshape data
# Convert the data to torch tensors and create datasets
train_dataset = TensorDataset(train_images.view(-1, input_size).float(), train_labels.long())
val_dataset = TensorDataset(val_images.view(-1, input_size).float(), val_labels.long())

# load data (select mini-batch=64)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  


# Train the model
Accuracy = []
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate the model
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
        
        accuracy = total_correct / total_samples
        Accuracy.append(accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')