import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


#Current configurations, might be adjusted after I gather final data:
N_CPG_SITES   = 10    # number of OXTR CpG sites measured
N_CLINICAL    = 2     # extra features: age (normalized), sex (0/1)
N_FEATURES    = N_CPG_SITES + N_CLINICAL
HIDDEN_SIZE   = 32   
DROPOUT       = 0.3
LR            = 1e-3
EPOCHS        = 50
BATCH_SIZE    = 16
SEED          = 42


class OXTRDataset(Dataset):
    """
    Each sample = one person.
    Features : [cpg1, cpg2, ..., cpg10, age_norm, sex]
                - CpG values are methylation % (0-100), normalized to [0,1]
                - age_norm: age divided by 100
                - sex: 0=female, 1=male
    Labels   : 0=healthy control, 1=OCD patient
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32) 
        self.y = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#Model arcitechture:

class OXTRModel(nn.Module):
    """

    Architecture:
        Input(12) → Linear(32) → ReLU → Dropout
                  → Linear(16) → ReLU → Dropout
                  → Linear(2)  → output logits
    """
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_SIZE, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(16, 2),   #2 outputs will be generated : [control_score, OCD_score]
        )

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        """Return probability of OCD (class 1)"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)[:, 1]


#Traning the model:
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            probs = torch.softmax(logits, dim=1)[:, 1] #convert logits to probabilities and keep probability of class 1 (OCD)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)   #calibrating the loss per sample
    preds    = (np.array(all_probs) >= 0.5).astype(int)
    acc      = accuracy_score(all_labels, preds)
    auroc    = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, acc, auroc, np.array(all_probs), np.array(all_labels)


#Plot of methylation levels and training curves: 

def plot_training(train_losses, val_losses, val_aurocs):
    """Plot learning curves — essential for a student project report."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train loss', color='steelblue')
    ax1.plot(val_losses,   label='Val loss',   color='coral')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and validation loss')
    ax1.legend()

    ax2.plot(val_aurocs, color='teal')
    ax2.axhline(0.5, linestyle='--', color='gray', label='Random chance')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUROC')
    ax2.set_title('Validation AUROC over training')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved to training_curves.png")


def plot_methylation_comparison(features, labels):
    """
    Visualizing the key finding: OCD patients have HIGHER OXTR methylation.
    This will match Bey et al. 2022 paper.
    """
    ocd_meth  = features[labels == 1, :N_CPG_SITES].mean(axis=1) * 100
    ctrl_meth = features[labels == 0, :N_CPG_SITES].mean(axis=1) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([ctrl_meth, ocd_meth], labels=['Control', 'OCD'])
    ax.set_ylabel('Mean OXTR methylation (%)')
    ax.set_title('OXTR methylation: OCD vs Control\n(replicating Bey et al. 2022)')
    ax.set_facecolor('#f8f8f8')
    plt.tight_layout()
    plt.savefig('oxtr_methylation_comparison.png', dpi=150)
    print("Methylation comparison saved to oxtr_methylation_comparison.png")


#Generating data until real data can be obtained

def generate_synthetic_data(n_samples=200, seed=42):
    """
    Simulate OXTR methylation data based on published findings.

    Key biological signal we're encoding (from Bey et al. 2022):
      - OCD patients: around 7-8% methylation at key CpG sites
      - Controls: around 4-5% methylation at key CpG sites
      - Values are percentages

    """
    rng = np.random.default_rng(seed)
    n_ocd  = n_samples // 2
    n_ctrl = n_samples - n_ocd

    # Controls have lower methylation (4-5%)
    ctrl_cpg = rng.normal(loc=0.045, scale=0.015, size=(n_ctrl, N_CPG_SITES))

    # OCD patients havehigher methylation (7-8%) 
    ocd_cpg  = rng.normal(loc=0.075, scale=0.018, size=(n_ocd,  N_CPG_SITES))

    # Clip to valid [0,1] range
    ctrl_cpg = np.clip(ctrl_cpg, 0, 1)
    ocd_cpg  = np.clip(ocd_cpg,  0, 1)

    # Clinical features are age (normalized), sex (binary), plan to integrate a different approach in next iterations
    ctrl_clinical = np.column_stack([
        rng.uniform(20, 60, n_ctrl) / 100,  # age normalized
        rng.integers(0, 2, n_ctrl),          # sex
    ])
    ocd_clinical = np.column_stack([
        rng.uniform(20, 60, n_ocd) / 100,
        rng.integers(0, 2, n_ocd),
    ])

    features = np.vstack([
        np.hstack([ctrl_cpg, ctrl_clinical]),
        np.hstack([ocd_cpg,  ocd_clinical]),
    ]).astype(np.float32)

    labels = np.array([0] * n_ctrl + [1] * n_ocd)
    return features, labels


def run_pipeline():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Synthetic data for this iteration
    features, labels = generate_synthetic_data(n_samples=200)
    print(f"Dataset: {len(labels)} samples  |  "
          f"OCD: {labels.sum()}  |  Control: {(labels==0).sum()}")
    print(f"Features per sample: {features.shape[1]} "
          f"({N_CPG_SITES} CpG sites + {N_CLINICAL} clinical)\n")

    # Show the methylation signal 
    plot_methylation_comparison(features, labels)

    # train validation split
    dataset = OXTRDataset(features, labels)
    n_val   = int(0.2 * len(dataset)) #reserved 20% of the samples for validation
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    
    model     = OXTRModel(n_features=N_FEATURES).to(device) #initialize the neural network
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  "
          f"{'Accuracy':>9}  {'AUROC':>7}")
    print("─" * 52)

    #Training loop:
    train_losses, val_losses, val_aurocs = [], [], []

    for epoch in range(1, EPOCHS + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc, v_auroc, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_aurocs.append(v_auroc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {t_loss:>10.4f}  {v_loss:>9.4f}  "
                  f"{v_acc:>9.4f}  {v_auroc:>7.4f}")

    # Evaluation:
    print("\n── Final evaluation on validation set ──")
    _, _, _, probs, true_labels = evaluate(
        model, val_loader, criterion, device)
    preds = (probs >= 0.5).astype(int)
    print(classification_report(true_labels, preds,
                                 target_names=["Control", "OCD"]))

    plot_training(train_losses, val_losses, val_aurocs)

    torch.save(model.state_dict(), "oxtr_ocd_model.pt")
    print("Model saved to oxtr_ocd_model.pt")

    return model
run_pipeline()
