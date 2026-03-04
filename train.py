from matplotlib import pyplot as plt


def train_model(model, train_loader, criterion, optimizer):
    print("Starting training...")
    num_epochs = 300
    best_loss = float('inf')
    patience = 10
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.long().squeeze())  # Convert to long and remove extra dimension
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        if epoch % 5 == 0:
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # create a plot of training loss after every 5 epochs but just add the points to the plot and not show the plot until the end of training
        if epoch % 5 == 0:
            plt.scatter(epoch, epoch_loss, color='blue')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
    # Show the training loss plot after training is complete
    plt.show()
