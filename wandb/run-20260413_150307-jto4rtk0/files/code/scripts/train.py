import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from src.utils.io import save_model

@hydra.main(version_base=None, config_path="../configs", config_name="train_n5s_snn")
def main(cfg: DictConfig):

    # Setup    
    if cfg.seed:
        torch.manual_seed(cfg.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    logger = instantiate(cfg.logger)
    logger.setup()

    # Initialise model, dataset, dataloaders, criterion, optimizer 
    dataset = instantiate(cfg.data)

    model = instantiate(
        cfg.model,
        input_size=dataset.X_shape[1],
        output_size=dataset.y_shape[1],
    ).to(device)

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
        trainer.train()
        
        final_acc, final_loss = trainer.test()
        print(f"\nFinal Loss: {final_loss:.4f}, Final Accuracy: {final_acc:.4f}")
        
        # Save the model
        save_model(cfg, model, logger)
        
    finally:
        # Ensure we finish the logger even if training fails
        logger.finish()

if __name__ == "__main__":
    main()