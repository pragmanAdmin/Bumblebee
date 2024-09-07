import torch

def train_model(model, train_dataloader, val_dataloader=None, epochs=5, learning_rate=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss}")

        if val_dataloader:
            val_loss, val_accuracy = evaluate_model(model, val_dataloader, device)
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    return model

def evaluate_model(model, dataloader, device='cpu'):
    model = model.to(device)
    model.eval()

    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / total_predictions

    return avg_loss, accuracy.item()