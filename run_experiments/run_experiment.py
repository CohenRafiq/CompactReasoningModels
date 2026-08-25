from compactreasoningmodels.experiments.noisy_similarity import NoisySimilarityExperiment

def main():
    experiment = NoisySimilarityExperiment(
        solvers=["arc_consistency", "local_min_violations", "global_min_violations", "discrete_genetic", "model_solver"],
        metrics=["MSE", "Pearson Correlation", "SSIM"],
        dataset_path="raw/nonogram_5x5_small.jsonl",
        target_shape=(5, 5)
    )
    results = experiment.run_experiment(
        num_samples=2,
        initial_grid="random",
        solver_steps={"arc_consistency": 1, "local_min_violations": 10, "global_min_violations": 10, "discrete_genetic": 30, "model_solver": 6},
        noise_level=0.3
    )
    for solver_name, df in results.items():
        print(f"\nResults for {solver_name}:")
        print(df)

if __name__ == "__main__":
    main()