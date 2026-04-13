import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch

@hydra.main(version_base=None, config_path="../configs", config_name="train_n5s_snn")
def main(cfg: DictConfig):

    # Setup
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.seed:
        torch.manual_seed(cfg.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    logger = instantiate(cfg.logger)
    logger.setup()

    # Initialise model, dataset, dataloaders, criterion, optimizer 
    model = instantiate(cfg.model).to(device)

    dataset = instantiate(cfg.data)
    
    train_size = int(cfg.split.get("train_ratio", 0.8) * len(dataset))
    test_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(cfg.split.get("generator_seed", 42))
    train_dataset, test_dataset = instantiate(
        cfg.split,
        dataset=dataset,
        lengths=[train_size, test_size],
        generator=generator
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.trainer.batch_size, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg.trainer.batch_size, 
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
        
        final_acc = trainer.test_accuracy()
        print(f"Final Test Accuracy: {final_acc:.4f}")
        
        # Save model
        model_path = "best_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.log_model(model_path, name="nonogram-solver")
        
    finally:
        # Ensure we finish the logger even if training fails
        logger.finish()

if __name__ == "__main__":
    main()