# def brier_residuals(discrete_step: np.ndarray, heatmap_step: np.ndarray) -> float:
#     brier = np.sum((discrete_step - heatmap_step) ** 2)
#     expected = np.sum(heatmap_step * (1 - heatmap_step))
#     return brier - expected

        # heatmap_steps = {name: solver.heatmap_step(initial_grid) for name, solver in initialised_solvers.items()}
        # discrete_steps = {name: solver.step(initial_grid) for name, solver in initialised_solvers.items()}

        # for heat_name, sample_name in itertools.product(heatmap_methods, sample_methods):
        #     heatmap_step = heatmap_steps[heat_name]
        #     discrete_step = discrete_steps[sample_name]

        #     residual = brier_residuals(discrete_step, heatmap_step)
        #     residuals[(heat_name, sample_name)].append(residual)

    # for (heat_name, sample_name), residual_list in residuals.items():
    #     result = lowess(heatmap_steps[heat_name], discrete_steps[sample_name], frac=0.1)
    #     print(f"  {heat_name} vs {sample_name}: {result}")

    # residuals = {(heat, sample): [] for heat, sample in itertools.product(heatmap_methods, sample_methods)}


    # heatmap_steps = {name: [] for name in heatmap_methods}
    # discrete_steps = {name: [] for name in sample_methods}

    # for heat_name, sample_name in itertools.product(heatmap_methods, sample_methods):
    #     heatmap_step = np.array(heatmap_steps[heat_name])
    #     discrete_step = np.array(discrete_steps[sample_name])

    #     print(f"{heat_name} vs {sample_name}:")
    #     print("  ", kernel_calibration_stat(heatmap_step, discrete_step, h=0.1))