import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from src.utils.io import save_model
from src.training.gradient_analyser import GradientAnalyser

@hydra.main(version_base=None, config_path="../configs", config_name="train_n15")
def main(cfg: DictConfig):

    if cfg.seed:
        torch.manual_seed(cfg.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    logger = instantiate(cfg.logger)
    logger.setup(cfg)

    # Initialise model, dataset, dataloaders, criterion, optimizer 
    dataset = instantiate(cfg.data)

    model = instantiate(
        cfg.model,
        input_size=dataset.input_shape,
        output_size=dataset.target_shape,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    train_size = int(cfg.split.train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Create generator and split
    generator = torch.Generator().manual_seed(cfg.split.seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size],
        generator=generator
    )
    
    train_loader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        shuffle=True
    )

    test_loader = instantiate(
        cfg.dataloader,
        dataset=test_dataset,
        shuffle=False
    )
    
    criterion = instantiate(cfg.criterion)
    
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    scheduler = instantiate(cfg.scheduler, optimizer=optimizer) if cfg.get("scheduler") else None

    gradient_analyser = GradientAnalyser(model, criterion)
    sample_inputs = next(iter(train_loader))
    gradient_analyser.run_full_report(sample_inputs)
    
    # Train and evaluate the model
    try:
        trainer = instantiate(
            cfg.trainer,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logger=logger
        )
        # During training — track drift
        gradient_analyser.register_hooks()
        trainer.train()
        gradient_analyser.remove_hooks()

        
        print(trainer.test())
        
        # Save the model
        save_model(cfg, model, logger)

        # After training — review what happened
        gradient_analyser.print_summary()
        
    finally:
        # Ensure we finish the logger even if training fails
        logger.finish()

if __name__ == "__main__":
    main()