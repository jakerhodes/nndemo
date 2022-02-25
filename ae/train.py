from matplotlib.pyplot import xticks
import torch
from tqdm import tqdm
from evaluate import evaluate

def train_classifier(model, epochs, train_loader, optimizer, criterion,
                val_loader = None, scheduler = None, device = 'cpu', track_loss = False):

    losses = []
    accuracies = []

    if val_loader is not None:
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

        if scheduler is not None:
            scheduler.step()

        # Calculate and append accuracies
        accuracy = correct / count
        accuracy = accuracy.detach().cpu().numpy()
        accuracies.append(accuracy)
        losses.append(running_loss)

        if (epoch + 1) % 5 == 0:
            print('Loss: ', running_loss, 'Accuracy: ', accuracy)

        if val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if (epoch + 1) % 5 == 0:
                print('Val Loss: ', val_loss, 'Val Accuracy: ', val_accuracy)

    if track_loss:
        return model, losses, accuracies, val_losses, val_accuracies
    else:
        return model


def train_grae(model, epochs, train_loader, optimizer, criterion, alpha = 0.5,
                val_loader = None, scheduler = None, device = 'cpu', track_loss = False):

    losses = []

    if val_loader is not None:
        val_losses = []

    for _, epoch in enumerate(range(epochs)):


        print('Epoch {}'.format(epoch + 1))
        model = model.train()
        model = model.to(device)

        count = 0
        running_loss = 0.0
        
        # Main Training Loop
        for x, embedding in tqdm(train_loader):

            x, embedding = x.to(device), embedding.to(device)

            optimizer.zero_grad()

            recon = model(x)
            z     = model.Encoder(x)

            loss_recon = criterion(recon, x)
            loss_emb   = criterion(z, embedding)
            # loss       = loss_recon + lamb * loss_emb
            loss       = alpha * loss_recon + (1 - alpha) * loss_emb

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += len(x)

        if scheduler is not None:
            scheduler.step()

        losses.append(running_loss)

        if (epoch + 1) % 5 == 0:
            print('Loss: ', running_loss)

        if val_loader is not None:
            val_loss, _ = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)

            if (epoch + 1) % 5 == 0:
                print('Val Loss: ', val_loss)

    if track_loss:
        return model, losses, val_losses
    else:
        return model





def rf_ae(model, epochs, train_loader, optimizer, criterion, lamb = 0,
                val_loader = None, scheduler = None, device = 'cpu', track_loss = False):

    losses = []

    if val_loader is not None:
        val_losses = []

    for _, epoch in enumerate(range(epochs)):

        print('Epoch {}'.format(epoch + 1))
        model = model.train()
        model = model.to(device)

        count = 0
        running_loss = 0.0
        
        # Main Training Loop
        for x, prox in tqdm(train_loader):

            x, prox = x.to(device), prox.to(device)

            optimizer.zero_grad()

            recon = model(x)
            z     = model.Encoder(x)

            loss_recon = criterion(recon, x)
            loss_emb   = criterion(z, embedding)
            loss       = loss_recon + lamb * loss_emb

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += len(x)

        if scheduler is not None:
            scheduler.step()

        losses.append(running_loss)

        if (epoch + 1) % 5 == 0:
            print('Loss: ', running_loss)

        if val_loader is not None:
            val_loss, _ = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)

            if (epoch + 1) % 5 == 0:
                print('Val Loss: ', val_loss)

    if track_loss:
        return model, losses, [val_losses if val_loader is not None else _]
    else:
        return model



def train_sae(model, epochs, train_loader, optimizer, criterion_pred, criterion_recon, alpha = .5,
                val_loader = None, scheduler = None, device = 'cpu', track_loss = False):

    losses = []
    accuracies = []

    if val_loader is not None:
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

            recon, pred = model(img)

            # Need separate for recon and for prediction
            loss_pred = criterion_pred(pred, label)
            loss_recon = criterion_recon(recon, img)

            loss = alpha * loss_recon + (1 - alpha) * loss_pred
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            correct += torch.sum(torch.argmax(pred, dim = 1) == label)
            count += len(img)

        if scheduler is not None:
            scheduler.step()

        # Calculate and append accuracies
        accuracy = correct / count
        accuracy = accuracy.detach().cpu().numpy()
        accuracies.append(accuracy)
        losses.append(running_loss)

        if (epoch + 1) % 5 == 0:
            print('Loss: ', running_loss, 'Accuracy: ', accuracy)

        if val_loader is not None:
            val_loss, val_accuracy = evaluate(model.Classifier, val_loader, criterion_pred, device) #Too specific
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if (epoch + 1) % 5 == 0:
                print('Val Loss: ', val_loss, 'Val Accuracy: ', val_accuracy)

    if track_loss:
        return model, losses, accuracies, val_losses, val_accuracies
    else:
        return model

def train_ae(model, epochs, train_loader, optimizer, criterion,
                val_loader = None, scheduler = None, device = 'cpu', track_loss = False):

    losses = []
    accuracies = []

    if val_loader is not None:
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

            recon, pred = model(img)

            # Need separate for recon and for prediction
            loss_pred = criterion_pred(pred, label)
            loss_recon = criterion_recon(recon, img)

            loss = alpha * loss_recon + (1 - alpha) * loss_pred
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            correct += torch.sum(torch.argmax(pred, dim = 1) == label)
            count += len(img)

        if scheduler is not None:
            scheduler.step()

        # Calculate and append accuracies
        accuracy = correct / count
        accuracy = accuracy.detach().cpu().numpy()
        accuracies.append(accuracy)
        losses.append(running_loss)

        if (epoch + 1) % 5 == 0:
            print('Loss: ', running_loss, 'Accuracy: ', accuracy)

        if val_loader is not None:
            val_loss, val_accuracy = evaluate(model.Classifier, val_loader, criterion_pred, device) #Too specific
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if (epoch + 1) % 5 == 0:
                print('Val Loss: ', val_loss, 'Val Accuracy: ', val_accuracy)

    if track_loss:
        return model, losses, accuracies, val_losses, val_accuracies
    else:
        return model