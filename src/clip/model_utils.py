
import torch
import clip
from torchvision import transforms

def load_clip_model(model_path, device='cpu'):
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, preprocess

def setup_model_and_optimizer(clip_model='ViT-B/32', learning_rate=5e-5, weight_decay=0.2, device='cpu'):
    model, preprocess = clip.load(clip_model, device=device)
    transform = transforms.Compose([preprocess])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=weight_decay
    )
    return model, transform, optimizer
