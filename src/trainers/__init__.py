from .trainer_with_attributes import CapsNetTrainer

def get_trainer(config, model, data_loader, loss_criterion):
    batch_size = data_loader.batch_size 
    
    match config['name']: 
        case 'TrainerWithAttributes':
            return CapsNetTrainer(
                loaders=data_loader,
                batch_size=batch_size,
                learning_rate=config['learning_rate'],
                lr_decay=config['lr_decay'],
                network=model,
                criterion=loss_criterion
            )
        case _:
            raise NotImplementedError(f'Unknown trainer: {config['name']}')