import torch

def display_grid(ax, grid, clues=None, title=''):
    ax.imshow(grid.cpu().numpy(), cmap='gray_r', interpolation='nearest')
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1, clip_on=False)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color='black', linewidth=1, clip_on=False)
    ax.axis('off')
    if clues is not None:
        for i, row_clue in enumerate(clues[0]):
            clue_str = ",".join(str(int(num)) for num in row_clue if num > 0) or "0"
            ax.text(-0.6, i, clue_str, va='center', ha='right', fontsize=12)
        for j, col_clue in enumerate(clues[1]):
            clue_str = ",".join(str(int(num)) for num in col_clue if num > 0) or "0"
            ax.text(j, -0.6, clue_str, va='bottom', ha='center', fontsize=12)
    ax.set_title(title, fontsize=14)

def display_blended(ax, grid, title=''):
    smooth = torch.softmax(grid, dim=0)
    smooth = smooth.permute(1, 2, 0)
    base_colours = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    flat_weights = smooth.reshape(-1, 3)
    blended_flat = flat_weights @ base_colours
    blended = blended_flat.reshape(smooth.shape)
    img = blended.detach().cpu().numpy()
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([i - 0.5 for i in range(grid.shape[1] + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(grid.shape[0] + 1)], minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.axis('off')
    ax.set_title(title, fontsize=14)
