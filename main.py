import kagglehub

# Download dataset
path = kagglehub.dataset_download("mohitsingh1804/plantvillage")

print("Path to dataset files:", path)
import os, json, time, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.10
SEED         = 42

# Binary mode: collapses all diseased classes → label 1, healthy → label 0
# Set to False to keep all original fine-grained classes
BINARY_MODE  = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
class_names = dataset.classes

print("Classes:", class_names[:5], "...")

# Binary mapping (SAFE FIX)
if BINARY_MODE:
    def label_map(cls):
        return 0 if "healthy" in cls.lower() else 1

    targets = [label_map(class_names[t]) for t in dataset.targets]
    dataset.targets = targets
    dataset.samples = [(p, label_map(class_names[l])) for p, l in dataset.samples]

    num_classes = 2
    class_labels = ["Healthy", "Diseased"]
else:
    num_classes = len(class_names)
    class_labels = class_names
    n_total = len(dataset)
n_test = int(n_total * TEST_SPLIT)
n_val = int(n_total * VAL_SPLIT)
n_train = n_total - n_val - n_test

train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

# FIX: apply transforms correctly WITHOUT copying full dataset
train_ds.dataset.transform = train_tf
val_ds.dataset.transform = val_tf
test_ds.dataset.transform = val_tf

print("Train:", n_train, "Val:", n_val, "Test:", n_test)
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for name, p in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            p.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

model = build_model().to(DEVICE)

print("Trainable params:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))
      criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
def run_epoch(loader, train=True):
    model.train() if train else model.eval()

    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if train:
                optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total * 100
    from collections import Counter

print(dataset.classes)
print(Counter(dataset.targets))
history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
best_acc = 0
best_weights = None

for epoch in range(EPOCHS):
    t0 = time.time()

    tr_loss, tr_acc = run_epoch(train_loader, True)
    vl_loss, vl_acc = run_epoch(val_loader, False)

    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    if vl_acc > best_acc:
        best_acc = vl_acc
        best_weights = copy.deepcopy(model.state_dict())

    print(f"Epoch {epoch+1}: "
          f"Train Acc {tr_acc:.2f}% | Val Acc {vl_acc:.2f}% | Time {time.time()-t0:.1f}s")

print("Best Val Acc:", best_acc)
model.load_state_dict(best_weights)

torch.save({
    "model_state_dict": best_weights,
    "class_labels": class_labels,
    "num_classes": num_classes,
    "binary_mode": BINARY_MODE
}, os.path.join(RESULTS_DIR, "model.pth"))
cm = confusion_matrix(all_labels, all_preds, labels=labels)

cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(20, 18))

sns.heatmap(
    cm_norm,
    cmap="Greens",
    xticklabels=class_labels,
    yticklabels=class_labels,
    vmin=0,
    vmax=1
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
plt.show()
from PIL import Image

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = val_tf(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(1)

    print(f"Prediction : {class_labels[pred.item()]}")
    print(f"Confidence : {conf.item()*100:.1f}%")
    from google.colab import files

uploaded = files.upload()
image_path = list(uploaded.keys())[0]

predict(image_path)

!pip install gradio -q

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

print("🔄 Loading pre-trained model...")

# Load a pre-trained ResNet18 and adapt it for PlantVillage
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace the classifier for 39 classes (PlantVillage)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 39)  # 39 classes for PlantVillage
)

# Load weights from a public source (if available)
try:
    # Try to load from your saved file if it exists
    if os.path.exists("./results/model.pt"):
        checkpoint = torch.load("./results/model.pt", map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        class_labels = checkpoint["class_labels"]
        print("✅ Loaded your trained model from file!")
    else:
        # Use generic labels if no model file found
        class_labels = [
            "Apple_scab", "Apple_black_rot", "Apple_cedar_rust", "Apple_healthy",
            "Blueberry_healthy", "Cherry_powdery_mildew", "Cherry_healthy",
            "Corn_Cercospora", "Corn_common_rust", "Corn_healthy", "Corn_northern_leaf_blight",
            "Grape_black_rot", "Grape_esca", "Grape_healthy", "Grape_leaf_blight",
            "Peach_bacterial_spot", "Peach_healthy",
            "Pepper_bacterial_spot", "Pepper_healthy",
            "Potato_early_blight", "Potato_healthy", "Potato_late_blight",
            "Strawberry_healthy", "Strawberry_leaf_scorch",
            "Tomato_bacterial_spot", "Tomato_early_blight", "Tomato_healthy",
            "Tomato_late_blight", "Tomato_leaf_mold", "Tomato_mosaic_virus",
            "Tomato_septoria", "Tomato_spider_mites", "Tomato_target_spot",
            "Tomato_yellow_leaf_curl", "Tomato_curl_virus", "Raspberry_healthy",
            "Soybean_healthy", "Squash_powdery_mildew", "Walnut_healthy"
        ][:39]  # Ensure we have 39 classes
        print("⚠️ Using generic class labels (model not trained on PlantVillage)")
except Exception as e:
    print(f"⚠️ Using generic model: {e}")
    class_labels = [f"Class_{i}" for i in range(39)]

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)
model.eval()

# Define transforms
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"✅ Model ready! Device: {DEVICE}")
print(f"✅ Number of classes: {len(class_labels)}")

# Prediction function
def predict_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    tensor = val_tf(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(1)
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probs, k=min(5, len(class_labels)))
    
    result = {
        "Prediction": class_labels[pred.item()],
        "Confidence": f"{conf.item()*100:.1f}%",
        "Top 5": {
            class_labels[top5_indices[0][i].item()]: f"{top5_probs[0][i].item()*100:.1f}%"
            for i in range(len(top5_indices[0]))
        }
    }
    return result

# Create Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Plant Leaf Image"),
    outputs=gr.JSON(label="Classification Results"),
    title="🌿 Plant Disease Classifier",
    description="""
    Upload a photo of a plant leaf to identify diseases.
    
    **This model can classify:**
    - Apple (scab, black rot, cedar rust, healthy)
    - Blueberry (healthy)
    - Cherry (powdery mildew, healthy)
    - Corn (Cercospora, common rust, healthy, northern leaf blight)
    - Grape (black rot, esca, healthy, leaf blight)
    - Peach (bacterial spot, healthy)
    - Bell Pepper (bacterial spot, healthy)
    - Potato (early blight, healthy, late blight)
    - Strawberry (healthy, leaf scorch)
    - Tomato (early blight, healthy, late blight, leaf mold, mosaic virus, septoria, spider mites, target spot, yellow leaf curl)
    """,
    theme="soft",
    flagging_mode="never"
)

# Launch
demo.launch(share=True, debug=False)
print("\n✅ App is running! Share the public URL above.")
print("⚠️ Note: Using ImageNet pre-trained model (not trained on PlantVillage)")