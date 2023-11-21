from data import RALO_Dataset
from net import Model
import pytorch_lightning as pl
from torch.utils.data import DataLoader

def main():
    # Initialize the training dataset
    train_dataset = RALO_Dataset(
        imgpath="data/RALO-Dataset/CXR_images_scored/",
        csvpath="data/RALO-Dataset/ralo-dataset-metadata.csv",
        subset="train"
    )

    # Initialize the validation dataset
    val_dataset = RALO_Dataset(
        imgpath="data/RALO-Dataset/CXR_images_scored/",
        csvpath="data/RALO-Dataset/ralo-dataset-metadata.csv",
        subset="val"
    )

    # Create DataLoaders for the training and validation datasets
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)

    # Initialize the model
    model = Model()

    # Setup a PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=10)

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the model
    model_path = "saved_models/model.ckpt"
    trainer.save_checkpoint(model_path)

if __name__ == '__main__':
    main()
