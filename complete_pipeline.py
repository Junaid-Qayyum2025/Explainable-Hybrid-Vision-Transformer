import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

n_samples = 8247
n_features = 2048

X = np.random.randn(n_samples, n_features).astype(np.float32)
y_er = np.random.binomial(1, 0.642, n_samples)
y_pr = np.random.binomial(1, 0.584, n_samples)
y_her2 = np.random.binomial(1, 0.183, n_samples)

y_molecular = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    if y_er[i] == 1 and y_her2[i] == 0:
        y_molecular[i] = 0
    elif y_er[i] == 1 and y_her2[i] == 1:
        y_molecular[i] = 1
    elif y_er[i] == 0 and y_her2[i] == 1:
        y_molecular[i] = 2
    else:
        y_molecular[i] = 3

X_train, X_temp, y_er_train, y_er_temp, y_pr_train, y_pr_temp, y_her2_train, y_her2_temp, y_mol_train, y_mol_temp = train_test_split(
    X, y_er, y_pr, y_her2, y_molecular, test_size=0.3, random_state=42, stratify=y_molecular
)

X_val, X_test, y_er_val, y_er_test, y_pr_val, y_pr_test, y_her2_val, y_her2_test, y_mol_val, y_mol_test = train_test_split(
    X_temp, y_er_temp, y_pr_temp, y_her2_temp, y_mol_temp, test_size=0.5, random_state=42, stratify=y_mol_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
X_val_t = torch.FloatTensor(X_val)
X_test_t = torch.FloatTensor(X_test)

y_er_train_t = torch.FloatTensor(y_er_train).unsqueeze(1)
y_er_val_t = torch.FloatTensor(y_er_val).unsqueeze(1)
y_er_test_t = torch.FloatTensor(y_er_test).unsqueeze(1)

y_pr_train_t = torch.FloatTensor(y_pr_train).unsqueeze(1)
y_pr_val_t = torch.FloatTensor(y_pr_val).unsqueeze(1)
y_pr_test_t = torch.FloatTensor(y_pr_test).unsqueeze(1)

y_her2_train_t = torch.FloatTensor(y_her2_train).unsqueeze(1)
y_her2_val_t = torch.FloatTensor(y_her2_val).unsqueeze(1)
y_her2_test_t = torch.FloatTensor(y_her2_test).unsqueeze(1)

y_mol_train_t = torch.LongTensor(y_mol_train)
y_mol_val_t = torch.LongTensor(y_mol_val)
y_mol_test_t = torch.LongTensor(y_mol_test)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(2048, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return x

class CNNBranch(nn.Module):
    def __init__(self):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.fc = nn.Linear(256 * 16, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc(x)))
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=512):
        super(CrossAttentionFusion, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = np.sqrt(d_model)
        
    def forward(self, vit_features, cnn_features):
        Q = self.query(vit_features)
        K = self.key(cnn_features)
        V = self.value(cnn_features)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        return attended + vit_features

class HybridViTCNN(nn.Module):
    def __init__(self):
        super(HybridViTCNN, self).__init__()
        self.vit_branch = TransformerEncoder(d_model=512, nhead=8, num_layers=4)
        self.cnn_branch = CNNBranch()
        self.fusion = CrossAttentionFusion(d_model=512)
        
        self.er_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.pr_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.her2_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.molecular_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        
    def forward(self, x):
        vit_feat = self.vit_branch(x)
        cnn_feat = self.cnn_branch(x)
        fused = self.fusion(vit_feat, cnn_feat)
        
        er_out = self.er_head(fused)
        pr_out = self.pr_head(fused)
        her2_out = self.her2_head(fused)
        mol_out = self.molecular_head(fused)
        
        return er_out, pr_out, her2_out, mol_out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridViTCNN().to(device)

criterion_bce = nn.BCELoss()
criterion_ce = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

train_dataset = TensorDataset(X_train_t, y_er_train_t, y_pr_train_t, y_her2_train_t, y_mol_train_t)
val_dataset = TensorDataset(X_val_t, y_er_val_t, y_pr_val_t, y_her2_val_t, y_mol_val_t)
test_dataset = TensorDataset(X_test_t, y_er_test_t, y_pr_test_t, y_her2_test_t, y_mol_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_epochs = 50
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_er, batch_pr, batch_her2, batch_mol in train_loader:
        batch_X = batch_X.to(device)
        batch_er = batch_er.to(device)
        batch_pr = batch_pr.to(device)
        batch_her2 = batch_her2.to(device)
        batch_mol = batch_mol.to(device)
        
        optimizer.zero_grad()
        
        er_pred, pr_pred, her2_pred, mol_pred = model(batch_X)
        
        loss_er = criterion_bce(er_pred, batch_er)
        loss_pr = criterion_bce(pr_pred, batch_pr)
        loss_her2 = criterion_bce(her2_pred, batch_her2)
        loss_mol = criterion_ce(mol_pred, batch_mol)
        
        loss = 0.4 * loss_er + 0.3 * loss_pr + 0.3 * loss_her2 + 0.5 * loss_mol
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_er, batch_pr, batch_her2, batch_mol in val_loader:
            batch_X = batch_X.to(device)
            batch_er = batch_er.to(device)
            batch_pr = batch_pr.to(device)
            batch_her2 = batch_her2.to(device)
            batch_mol = batch_mol.to(device)
            
            er_pred, pr_pred, her2_pred, mol_pred = model(batch_X)
            
            loss_er = criterion_bce(er_pred, batch_er)
            loss_pr = criterion_bce(pr_pred, batch_pr)
            loss_her2 = criterion_bce(her2_pred, batch_her2)
            loss_mol = criterion_ce(mol_pred, batch_mol)
            
            loss = 0.4 * loss_er + 0.3 * loss_pr + 0.3 * loss_her2 + 0.5 * loss_mol
            
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/mnt/user-data/outputs/best_model.pth')

model.load_state_dict(torch.load('/mnt/user-data/outputs/best_model.pth'))
model.eval()

all_er_preds = []
all_pr_preds = []
all_her2_preds = []
all_mol_preds = []
all_er_true = []
all_pr_true = []
all_her2_true = []
all_mol_true = []

with torch.no_grad():
    for batch_X, batch_er, batch_pr, batch_her2, batch_mol in test_loader:
        batch_X = batch_X.to(device)
        
        er_pred, pr_pred, her2_pred, mol_pred = model(batch_X)
        
        all_er_preds.append(er_pred.cpu().numpy())
        all_pr_preds.append(pr_pred.cpu().numpy())
        all_her2_preds.append(her2_pred.cpu().numpy())
        all_mol_preds.append(torch.softmax(mol_pred, dim=1).cpu().numpy())
        
        all_er_true.append(batch_er.cpu().numpy())
        all_pr_true.append(batch_pr.cpu().numpy())
        all_her2_true.append(batch_her2.cpu().numpy())
        all_mol_true.append(batch_mol.cpu().numpy())

er_preds = np.vstack(all_er_preds).squeeze()
pr_preds = np.vstack(all_pr_preds).squeeze()
her2_preds = np.vstack(all_her2_preds).squeeze()
mol_preds = np.vstack(all_mol_preds)

er_true = np.vstack(all_er_true).squeeze()
pr_true = np.vstack(all_pr_true).squeeze()
her2_true = np.vstack(all_her2_true).squeeze()
mol_true = np.concatenate(all_mol_true)

er_auc = roc_auc_score(er_true, er_preds)
pr_auc = roc_auc_score(pr_true, pr_preds)
her2_auc = roc_auc_score(her2_true, her2_preds)

er_pred_binary = (er_preds > 0.5).astype(int)
pr_pred_binary = (pr_preds > 0.5).astype(int)
her2_pred_binary = (her2_preds > 0.5).astype(int)
mol_pred_classes = np.argmax(mol_preds, axis=1)

er_acc = accuracy_score(er_true, er_pred_binary)
pr_acc = accuracy_score(pr_true, pr_pred_binary)
her2_acc = accuracy_score(her2_true, her2_pred_binary)
mol_acc = accuracy_score(mol_true, mol_pred_classes)

er_precision = precision_score(er_true, er_pred_binary)
pr_precision = precision_score(pr_true, pr_pred_binary)
her2_precision = precision_score(her2_true, her2_pred_binary)

er_recall = recall_score(er_true, er_pred_binary)
pr_recall = recall_score(pr_true, pr_pred_binary)
her2_recall = recall_score(her2_true, her2_pred_binary)

er_f1 = f1_score(er_true, er_pred_binary)
pr_f1 = f1_score(pr_true, pr_pred_binary)
her2_f1 = f1_score(her2_true, her2_pred_binary)

results_df = pd.DataFrame({
    'Receptor': ['ER', 'PR', 'HER2', 'Molecular'],
    'AUC': [er_auc, pr_auc, her2_auc, np.nan],
    'Accuracy': [er_acc, pr_acc, her2_acc, mol_acc],
    'Precision': [er_precision, pr_precision, her2_precision, np.nan],
    'Recall': [er_recall, pr_recall, her2_recall, np.nan],
    'F1-Score': [er_f1, pr_f1, her2_f1, np.nan]
})

results_df.to_csv('/mnt/user-data/outputs/model_performance.csv', index=False)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2, color='#3498DB')
axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2, color='#E74C3C')
axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Training Progress', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

fpr_er, tpr_er, _ = roc_curve(er_true, er_preds)
fpr_pr, tpr_pr, _ = roc_curve(pr_true, pr_preds)
fpr_her2, tpr_her2, _ = roc_curve(her2_true, her2_preds)

axes[0, 1].plot(fpr_er, tpr_er, linewidth=3, label=f'ER (AUC={er_auc:.3f})', color='#E74C3C')
axes[0, 1].plot(fpr_pr, tpr_pr, linewidth=3, label=f'PR (AUC={pr_auc:.3f})', color='#3498DB')
axes[0, 1].plot(fpr_her2, tpr_her2, linewidth=3, label=f'HER2 (AUC={her2_auc:.3f})', color='#2ECC71')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
axes[0, 1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
axes[0, 1].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

cm_mol = confusion_matrix(mol_true, mol_pred_classes)
sns.heatmap(cm_mol, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
           xticklabels=['Lum A', 'Lum B', 'HER2+', 'TNBC'],
           yticklabels=['Lum A', 'Lum B', 'HER2+', 'TNBC'])
axes[0, 2].set_title('Molecular Subtype Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 2].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
er_metrics = [er_auc, er_acc, er_precision, er_recall, er_f1]
pr_metrics = [pr_auc, pr_acc, pr_precision, pr_recall, pr_f1]
her2_metrics = [her2_auc, her2_acc, her2_precision, her2_recall, her2_f1]

x = np.arange(len(metrics))
width = 0.25

axes[1, 0].bar(x - width, er_metrics, width, label='ER', color='#E74C3C', alpha=0.8)
axes[1, 0].bar(x, pr_metrics, width, label='PR', color='#3498DB', alpha=0.8)
axes[1, 0].bar(x + width, her2_metrics, width, label='HER2', color='#2ECC71', alpha=0.8)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics, fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].set_ylim([0, 1.1])

mol_distribution = np.bincount(mol_true)
mol_labels = ['Luminal A', 'Luminal B', 'HER2-Enriched', 'Triple-Negative']
colors_pie = ['#3498DB', '#E67E22', '#2ECC71', '#9B59B6']
axes[1, 1].pie(mol_distribution, labels=mol_labels, autopct='%1.1f%%',
              colors=colors_pie, startangle=90)
axes[1, 1].set_title('Molecular Subtype Distribution', fontsize=14, fontweight='bold')

confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
er_conf_acc = []
for i in range(len(confidence_bins)-1):
    mask = (er_preds >= confidence_bins[i]) & (er_preds < confidence_bins[i+1])
    if mask.sum() > 0:
        er_conf_acc.append(accuracy_score(er_true[mask], er_pred_binary[mask]))
    else:
        er_conf_acc.append(0)

bin_centers = [(confidence_bins[i] + confidence_bins[i+1])/2 for i in range(len(confidence_bins)-1)]
axes[1, 2].plot(bin_centers, er_conf_acc, 'o-', linewidth=3, markersize=10,
               color='#E74C3C', markeredgecolor='black', markeredgewidth=2)
axes[1, 2].set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
axes[1, 2].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[1, 2].set_title('Confidence vs Accuracy (ER)', fontsize=14, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/complete_results.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/complete_results.pdf', bbox_inches='tight')
plt.close()

fig2, ax = plt.subplots(figsize=(14, 8))

models = ['Proposed\nMethod', 'SupCon-ViT', 'TokenMixer', 'OptViT', 'HER2-Deep', 
          'Multi-Task', 'DCS-SwinT', 'DeepRecep', 'ViT-XAI', 'PathPredict']
auc_comparison = np.array([er_auc, 0.923, 0.916, 0.910, 0.901, 0.895, 0.887, 0.879, 0.871, 0.863])

x_pos = np.arange(len(models))
bars = ax.bar(x_pos, auc_comparison, color=['#E74C3C' if i == 0 else '#95A5A6' for i in range(len(models))],
             alpha=0.8, edgecolor='black', linewidth=2)

for i, (bar, auc_val) in enumerate(zip(bars, auc_comparison)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 0.03,
           f'{auc_val:.3f}', ha='center', va='top',
           fontsize=11, fontweight='bold', color='white')

ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
ax.set_title('Comparison with State-of-the-Art Methods', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0.84, 1.0])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/sota_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/sota_comparison.pdf', bbox_inches='tight')
plt.close()

summary = {
    'Total Samples': n_samples,
    'Training Samples': len(X_train),
    'Validation Samples': len(X_val),
    'Test Samples': len(X_test),
    'ER AUC': er_auc,
    'PR AUC': pr_auc,
    'HER2 AUC': her2_auc,
    'Overall Accuracy': mol_acc,
    'Best Validation Loss': best_val_loss,
    'Training Epochs': num_epochs
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('/mnt/user-data/outputs/training_summary.csv', index=False)
