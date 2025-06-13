
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def test_clip_model(model, test_loader, device='cpu', class_names=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='Testing'):
            imgs, labels = imgs.to(device), labels.to(device)
            feats = model.encode_image(imgs)
            logits = model.logit_scale.exp() * feats @ model.text_projection.t()
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum([p == t for p, t in zip(all_preds, all_labels)]) / len(all_preds)
    print(f'Accuracy: {acc:.4f}')

    if class_names:
        print(classification_report(all_labels, all_preds, target_names=class_names))
        plot_confusion_matrix(all_labels, all_preds, class_names)

def plot_confusion_matrix(true, pred, classes):
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
