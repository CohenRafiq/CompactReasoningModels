from compactreasoningmodels.experiments.heatmap_store import HeatmapStore
from compactreasoningmodels.datasets.nonogram_dataset import NonogramDataset
from torch.utils.data import DataLoader
from compactreasoningmodels.experiments.noisy_similarity import NoisySimilarityExperiment
from compactreasoningmodels.solving_traces.similarity_measures import MSE, MAE, HuberLoss, MeanCosineSimilarity, PearsonCorrelation, SSIMSimilarity, WassersteinSimilarity

dataset = NonogramDataset(
    data="data/raw/nonogram_5x5_tiny.jsonl",
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
print(f"Dataset size: {len(dataset)} samples")
heatmap_store = HeatmapStore(dataloader)
experiments = [
    ("arc_consistency", 1),
    ("arc_consistency", 2),
    ("local_min_violations", 1),
    ("global_min_violations", 10),
    ("discrete_genetic", 10),
    ("discrete_genetic", 30),
    ("model_solver", 1),
    ("model_solver", 8),
]
heatmap_store.add_heatmaps(experiments)
print("Heatmaps generated for the following experiments:")
heatmap_store.display_mse_loss()
print()
heatmap_store.display_clue_loss()

noisy_experiment = NoisySimilarityExperiment()
results = noisy_experiment.run_experiment(
    heatmaps = heatmap_store.heatmaps,
    metrics = [MSE, PearsonCorrelation, SSIMSimilarity],
    noise_level = 0.3
)
noisy_experiment.display_results(results)

