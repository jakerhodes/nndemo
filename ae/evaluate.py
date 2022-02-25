import torch

def evaluate(model, dataloader, criterion, device):
    model = model.eval()
    model = model.to(device)

    count = 0
    correct = 0.0
    running_loss = 0.0
    for _, (img, label) in enumerate(dataloader):

        label = label.to(device)
        pred = model(img.to(device))
        loss = criterion(pred, label.to(device))
        
        running_loss += loss.item()
        correct += torch.sum(torch.argmax(pred, dim = 1) == label)

        count += len(img)

    return running_loss, (correct / count).detach().cpu().numpy()

