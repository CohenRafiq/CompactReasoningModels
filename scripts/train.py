import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, get_class
import torch
from torch.utils.data import Subset, random_split
from src.utils.io import save_model

@hydra.main(version_base=None, config_path="../configs", config_name="train_n15")
def main(cfg: DictConfig):

    if cfg.seed:
        torch.manual_seed(cfg.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    logger = instantiate(cfg.logger)
    logger.setup(cfg)

    # Dataset, Model and dataloaders
    dataset = instantiate(cfg.data)

    model_class = get_class(cfg.model._target_)
    if model_class.require_flat_input and len(dataset.input_shape) > 1:
        dataset.flatten()
        
    model = instantiate(
        cfg.model,
        input_size=dataset.input_shape,
        output_size=dataset.target_shape,
    ).to(device)


    generator = torch.Generator().manual_seed(cfg.split.seed)
    # 
    shuffled_indices = torch.randperm(len(dataset), generator=generator).tolist()
    selected_indices = shuffled_indices[:1000]
    dataset = Subset(dataset, selected_indices)
    # 
    train_size = int(cfg.split.train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(
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

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Criterion, optimizer, scheduler
    criterion = instantiate(cfg.criterion)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer) if cfg.get("scheduler") else None

    try:
        # Train the model
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
        trainer.train()

        
        print(trainer.test())
        save_model(cfg, model, logger)
        
    finally:
        logger.finish()

if __name__ == "__main__":
    main()