
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_loaders(dataset, test_size=0.2, val_size=0.2, batch_size=16):
    indices = list(range(len(dataset)))
    labels = [dataset.samples[i][1] for i in indices]
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=labels, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, stratify=[labels[i] for i in train_val_idx], random_state=42)
    return (
        DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=4),
        DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=4)
    )

def train_clip_model(model, train_loader, val_loader, optimizer, num_epochs, device='cpu', save_path='model.pth'):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * num_epochs)
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0.0, 0

        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            feats = model.encode_image(imgs)
            logits = model.logit_scale.exp() * feats @ model.text_projection.t()
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(logits, 1)
            correct += torch.sum(preds == labels)

        scheduler.step()
        acc = correct.double() / len(train_loader.dataset)
        print(f'Loss: {total_loss / len(train_loader.dataset):.4f} | Acc: {acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print('Saved best model')

    print(f'Best validation accuracy: {best_acc:.4f}')
