import torch
from tqdm import tqdm
from eval import evaluate

def train_model(model, epochs, train_loader, val_loader, optimizer, criterion, device = 'cpu', track_loss = False):

    losses = []
    accuracies = []

    val_losses = []
    val_accuracies = []
    for _, epoch in enumerate(range(epochs)):


        print('Epoch {}'.format(epoch + 1))
        model = model.train()
        model = model.to(device)

        count = 0
        correct = 0.0
        running_loss = 0.0
        
        # Main Training Loop
        for img, label in tqdm(train_loader):

            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()

            pred = model(img)

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            correct += torch.sum(torch.argmax(pred, dim = 1) == label)
            count += len(img)

        accuracy = correct / count
        accuracy = accuracy.detach().cpu().numpy()

        accuracies.append(accuracy)
        losses.append(running_loss)

        if (epoch + 1) % 5 == 0:
            print('Loss: ', running_loss, 'Accuracy: ', accuracy)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if (epoch + 1) % 5 == 0:
            print('Val Loss: ', val_loss, 'Val Accuracy: ', val_accuracy)

    if track_loss:
        return model, losses, accuracies, val_losses, val_accuracies
    else:
        return model



