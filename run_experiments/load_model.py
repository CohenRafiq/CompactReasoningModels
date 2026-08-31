import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from compactreasoningmodels.models.recursive_gridmlp import RecursiveGridMLP
from compactreasoningmodels.utils.display import ascii_print_grid, display_grid, display_blended
from compactreasoningmodels.datasets.nonogram_dataset import NonogramDataset


# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecursiveGridMLP(
    hidden_size=256,
    num_layers=9,
    dropout=0.3,
    input_size=30,
    output_size=75
).to(device)
model_weight_path = "./models/jsonldataset/recursivegridmlp"
files = [f for f in os.listdir(model_weight_path) if f.endswith('.pt')]
max_num = max(int(f.split('.')[0]) for f in files)
model_weight_path = f"{model_weight_path}/{max_num:02d}.pt"

state_dict = torch.load(model_weight_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

# Load dataset
dataset = NonogramDataset(
    data="data/raw/nonogram_5x5_small.jsonl",
)

# Number of examples to display
num_examples = 5

# Create a figure with subplots (2 rows x 5 columns: top row for ground truth, bottom for predictions)
fig, axes = plt.subplots(2, num_examples, figsize=(3*num_examples, 6))

# Run model on first num_examples
model.eval()
with torch.no_grad():
    for i in range(num_examples):
        # Get ground truth grid
        input_tensor_full = dataset[i][0]
        target_grid = dataset[i][1].reshape(5, 5).numpy()
        
        # Display ground truth
        ax_gt = axes[0, i]
        display_grid(ax=ax_gt, grid=target_grid, title=f"GT #{i+1}")
        ax_gt.axis('off')
        
        # Run model prediction
        input_tensor = input_tensor_full.unsqueeze(0).to(device)
        output_tensor = model(input_tensor)
        output_grid = output_tensor.cpu().detach()[0].reshape(3, 5, 5)
        
        # Display model output
        ax_pred = axes[1, i]
        display_blended(ax=ax_pred, grid=output_grid, title=f"Pred #{i+1}")
        ax_pred.axis('off')

plt.tight_layout()
plt.savefig("./experiments/artifacts/first_5_examples.png", bbox_inches='tight', dpi=300)
plt.close()

print(f"Saved first {num_examples} examples to ./experiments/artifacts/first_5_examples.png")


fig, ax = plt.subplots(1, 9, figsize=(45, 5))
for i in range(9):
    output_tensor = model(input_tensor, layer_num=i + 1)
    output_grid = output_tensor.cpu().detach()[0].reshape(3, 5, 5)
    display_blended(ax=ax[i], grid=output_grid, title=f"Layer {i + 1}")
fig.savefig("./experiments/artifacts/model_output_layers.png", bbox_inches='tight', dpi=300)

