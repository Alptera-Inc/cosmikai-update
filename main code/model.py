import torch, torch.nn as nn

# GPU setup and diagnostics
if torch.cuda.is_available():
    print(f"cuda available {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
    # Set memory allocation strategy for better GPU utilization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    print("no cuda")

class SmallCNN(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):              # x: (B, L)
        x = x.unsqueeze(1)             # (B, 1, L)
        f = self.feature(x)            # (B, 128, 1)
        return self.head(f).squeeze(1) # (B,)