import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.cm as cm
import mlflow
import mlflow.pytorch

# CONFIGURATION
STUDENT_ID = "s3777006"

# Base path on SCRATCH for runtime
BASE_ROOT = Path(f"/scratch/{STUDENT_ID}")

DATA_ROOT = BASE_ROOT / "CP-CHILD"
DATASET = "CP-CHILD-A"  # or B

PRETRAINED_WEIGHTS_PATH = BASE_ROOT / "pretrained_weights" / "vgg19_bn_imagenet_weights.pth"
MLFLOW_ROOT = BASE_ROOT / "ML_Flow"

# Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
USE_PRETRAINED = True
MAX_SAMPLES = None  # Set to int for debugging with subset
SEED = 42


# REPRODUCIBILITY SETUP
def setup_reproducibility(seed):
    """Ensures deterministic behavior across all random operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Disable cuDNN optimization for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # DataLoader worker seeding
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    return seed_worker, g

seed_worker, generator = setup_reproducibility(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# MLflow tracking
MLFLOW_ROOT.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file:{MLFLOW_ROOT}")
mlflow.set_experiment(f"Experiment_A_{'Pretrained' if USE_PRETRAINED else 'Kaiming'}")


# GRAD-CAM IMPLEMENTATION
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        out[0, class_idx].backward()

        A = self.activations[0]
        G = self.gradients[0]
        weights = G.mean(dim=(1, 2))

        cam = (A * weights[:, None, None]).sum(dim=0)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        
        return cam


def make_overlay(img_tensor, cam, alpha=0.4):
    """Blend Grad-CAM heatmap with original image."""
    device = img_tensor.device
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    img = (img_tensor * std + mean).clamp(0, 1)
    
    # Resize CAM to image dimensions
    cam_resized = F.interpolate(
        cam[None, None], size=img.shape[-2:], mode="bilinear"
    )[0, 0].clamp(0, 1)
    
    # Apply JET colormap
    cam_np = cam_resized.detach().cpu().numpy()
    jet = cm.get_cmap("jet")(cam_np)[:, :, :3]
    heatmap = torch.from_numpy(jet).permute(2, 0, 1).to(device)
    
    # Blend
    overlay = (1 - alpha) * img + alpha * heatmap
    return overlay.clamp(0, 1).cpu()

# DATA LOADING
DATASET_ROOT = DATA_ROOT / DATASET
TRAIN_ROOT = DATASET_ROOT / "Train"
TEST_ROOT = DATASET_ROOT / "Test"

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

train_ds = datasets.ImageFolder(str(TRAIN_ROOT), transform=transform)
test_ds = datasets.ImageFolder(str(TEST_ROOT), transform=transform)

num_classes = len(train_ds.classes)
POS = train_ds.class_to_idx["Polyp"]
NEG = train_ds.class_to_idx["Non-Polyp"]

print(f"Train: {len(train_ds)} images | Test: {len(test_ds)} images")


def build_balanced_subset(dataset, pos_label, neg_label, max_samples):
    """Create balanced subset for debugging."""
    targets = dataset.targets
    pos_idx = [i for i, t in enumerate(targets) if t == pos_label]
    neg_idx = [i for i, t in enumerate(targets) if t == neg_label]
    
    n_per_class = max_samples // 2
    indices = pos_idx[:n_per_class] + neg_idx[:n_per_class]
    random.shuffle(indices)
    return indices


loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": 4,
    "pin_memory": (device.type == "cuda"),
    "worker_init_fn": seed_worker,
    "generator": generator,
}

if MAX_SAMPLES:
    train_subset = Subset(train_ds, build_balanced_subset(train_ds, POS, NEG, MAX_SAMPLES))
    test_subset = Subset(test_ds, build_balanced_subset(test_ds, POS, NEG, MAX_SAMPLES))
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)
    print(f"Using subset: {len(train_subset)} train, {len(test_subset)} test")
else:
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

#Basic VGG19 GAP architecture
class VGG19BN_GAP_Last2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Block 1
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2))

        # Block 2
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2))

        # Block 3
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.layer7 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.layer8 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2))

        # Block 4
        self.layer9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.layer10 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.layer11 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.layer12 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2))

        # Block 5
        self.layer13 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.layer14 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.layer15 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))

        # Final block → num_classes channels
        self.layer16 = nn.Sequential(
            nn.Conv2d(512, num_classes, 3, 1, 1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        for layer in [
            self.layer1, self.layer2, self.layer3, self.layer4, self.layer5,
            self.layer6, self.layer7, self.layer8, self.layer9, self.layer10,
            self.layer11, self.layer12, self.layer13, self.layer14, self.layer15,
            self.layer16,
        ]:
            x = layer(x)
        x = self.gap(x)
        return torch.flatten(x, 1)


# WEIGHT INITIALIZATION
def load_imagenet_pretrained(model, weights_path):
    """Load pretrained ImageNet weights, excluding final block."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
    
    print(f"Loading ImageNet weights from: {weights_path}")
    pretrained = torch.load(weights_path, map_location="cpu")
    new_state_dict = {}
    
    # Map torchvision VGG19-BN layer names to our custom names
    layer_mapping = {
        'features.0': 'layer1.0', 'features.1': 'layer1.1',
        'features.3': 'layer2.0', 'features.4': 'layer2.1',
        'features.7': 'layer3.0', 'features.8': 'layer3.1',
        'features.10': 'layer4.0', 'features.11': 'layer4.1',
        'features.14': 'layer5.0', 'features.15': 'layer5.1',
        'features.17': 'layer6.0', 'features.18': 'layer6.1',
        'features.20': 'layer7.0', 'features.21': 'layer7.1',
        'features.23': 'layer8.0', 'features.24': 'layer8.1',
        'features.27': 'layer9.0', 'features.28': 'layer9.1',
        'features.30': 'layer10.0', 'features.31': 'layer10.1',
        'features.33': 'layer11.0', 'features.34': 'layer11.1',
        'features.36': 'layer12.0', 'features.37': 'layer12.1',
        'features.40': 'layer13.0', 'features.41': 'layer13.1',
        'features.43': 'layer14.0', 'features.44': 'layer14.1',
        'features.46': 'layer15.0', 'features.47': 'layer15.1',
    }
    
    for old_key, new_key in layer_mapping.items():
        for param_type in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
            old_param = f"{old_key}.{param_type}"
            if old_param in pretrained:
                new_state_dict[f"{new_key}.{param_type}"] = pretrained[old_param]
    
    model.load_state_dict(new_state_dict, strict=False)
    print("Loaded pretrained layers 1-15")


def init_final_block(model):
    """Initialize layer16 with Kaiming normal."""
    for m in model.layer16.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# Initialize model
model = VGG19BN_GAP_Last2(num_classes=num_classes)

if USE_PRETRAINED:
    load_imagenet_pretrained(model, PRETRAINED_WEIGHTS_PATH)
    init_final_block(model)
    print("Using ImageNet pretrained + Kaiming for layer16")
else:
    # Full Kaiming initialization for training from scratch
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    print("Using full Kaiming initialization (no pretraining)")

model = model.to(device)
# TRAINING & EVALUATION
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
run_name = f"{DATASET}"
print(f"\nStarting training: {run_name}\n")

with mlflow.start_run(run_name=run_name):
    
    mlflow.log_params({
        "dataset": DATASET,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "momentum": MOMENTUM,
        "use_pretrained": USE_PRETRAINED,
        "max_samples": MAX_SAMPLES if MAX_SAMPLES else "full",
        "seed": SEED,
    })
    
    # Training loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            total += x.size(0)
        
        epoch_loss = running_loss / total
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss:.4f}")
        mlflow.log_metric("train_loss", epoch_loss, epoch)
    
    mlflow.pytorch.log_model(model, artifact_path="model")
    
    # Evaluation
    print("\nEvaluating...")
    TP = TN = FP = FN = 0
    example_store = {"TP": [], "TN": [], "FP": [], "FN": []}
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            y_cpu = y.cpu()
            
            TP += ((preds == POS) & (y_cpu == POS)).sum().item()
            TN += ((preds == NEG) & (y_cpu == NEG)).sum().item()
            FP += ((preds == POS) & (y_cpu == NEG)).sum().item()
            FN += ((preds == NEG) & (y_cpu == POS)).sum().item()
            
            # Store examples for Grad-CAM
            for i in range(x.size(0)):
                true_lbl = y_cpu[i].item()
                pred_lbl = preds[i].item()
                
                if true_lbl == POS and pred_lbl == POS:
                    group = "TP"
                elif true_lbl == NEG and pred_lbl == NEG:
                    group = "TN"
                elif true_lbl == NEG and pred_lbl == POS:
                    group = "FP"
                else:
                    group = "FN"
                
                if len(example_store[group]) < 2:
                    example_store[group].append((x[i].cpu(), true_lbl, pred_lbl))
    
    # Calculate metrics
    total_eval = TP + TN + FP + FN
    accuracy = (TP + TN) / total_eval
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * TPR) / (precision + TPR) if (precision + TPR) > 0 else 0
    
    print("\nResults:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (TPR): {TPR:.4f}")
    print(f"Specificity (TNR): {TNR:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    mlflow.log_metrics({
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy,
        "TPR": TPR,
        "TNR": TNR,
        "precision": precision,
        "f1": f1,
    })
    
    # Save confusion matrix
    cm_text = f"TN={TN} FP={FP}\nFN={FN} TP={TP}\n"
    cm_path = Path("confusion_matrix.txt")
    cm_path.write_text(cm_text)
    mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
    cm_path.unlink()
    
    # Grad-CAM visualization
    print("\nGenerating Grad-CAM examples...")
    
    gradcam = GradCAM(model, model.layer16)
    outdir = Path(f"/scratch/{STUDENT_ID}/gradcam_outputs/Experiment_A") / DATASET
    outdir.mkdir(parents=True, exist_ok=True)
    
    labels = test_ds.classes
    counter = 0
    
    model.eval()
    for group, samples in example_store.items():
        for img_tensor, true_lbl, pred_lbl in samples:
            img_dev = img_tensor.to(device)
            x_in = img_dev.unsqueeze(0)
            
            cam = gradcam.generate(x_in, pred_lbl)
            overlay = make_overlay(img_dev, cam)
            
            fname = outdir / f"{group}_true{labels[true_lbl]}_pred{labels[pred_lbl]}_{counter}.png"
            save_image(overlay, fname)
            counter += 1
    
    mlflow.log_artifacts(str(outdir), artifact_path="gradcam")
    print(f"Grad-CAM outputs saved to: {outdir}")

print(f"\nTraining complete. MLflow tracking at: {MLFLOW_ROOT}")
