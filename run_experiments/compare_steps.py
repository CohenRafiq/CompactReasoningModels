from compactreasoningmodels.experiments.heatmap_store import HeatmapStore
from compactreasoningmodels.datasets.jsonl import JsonlDataset
from torch.utils.data import DataLoader
from compactreasoningmodels.experiments.noisy_similarity import NoisySimilarityExperiment
from compactreasoningmodels.solving_traces.similarity_measures import MSE
from pandas import DataFrame as df

dataset = JsonlDataset(
    input_data="raw/nonogram_5x5_single.jsonl",
    target_data=None,
    target_shape=(5, 5)
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
print(f"Dataset size: {len(dataset)} samples")
heatmap_store = HeatmapStore(dataloader)
experiments = [("discrete_genetic", i) for i in range(0, 31, 3)] + \
                [("model_solver", i) for i in range(0, 10)] + \
                [("arc_consistency", i) for i in range(1, 5)] + \
                [("local_min_violations", i) for i in range(0, 21, 4)] + \
                [("global_min_violations", i) for i in range(0, 21, 4)]
heatmap_store.add_heatmaps(experiments)
print("Heatmaps generated for the following experiments:")
heatmap_store.display_mse_loss()
print()
heatmap_store.display_clue_loss()

noisy_experiment = NoisySimilarityExperiment()
results = noisy_experiment.run_experiment(
    heatmaps = heatmap_store.heatmaps,
    metrics = [MSE],
    noise_level = 0.0
)
result = noisy_experiment.display_results(results)
df.to_csv(result[0], "results/noisy_similarity_results_single_2.csv", index=False)

