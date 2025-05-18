import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import segmentation_models_pytorch as smp
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete

# Definizione del Dataset personalizzato per le immagini MRI
class MRIDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.tif')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*_mask.tif')))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale (1 channel)
        mask = np.array(mask) / 255.0  # Normalize mask to [0, 1]
        mask = torch.from_numpy(mask).float().unsqueeze(0) # Add channel dimension

        if self.transform:
            image = self.transform(image)
            # Potrebbe essere necessario applicare trasformazioni separate alla maschera
            mask = transforms.functional.resize(mask, image.shape[2:]) # Resize mask to image size

        return image, mask

# Funzione per l'allenamento del modello
def train_model(
    model, 
    dataloader, 
    criterion, 
    optimizer, 
    device,
):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Funzione per la valutazione del modello
def evaluate_model(
    model, 
    dataloader, 
    criterion, 
    device, 
    dice_metric, 
    iou_metric, 
    post_pred, 
    post_label,
):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()

            # Calcolo delle metriche
            outputs_sigmoid = torch.sigmoid(outputs)
            dice_metric(
                y_pred=post_pred(outputs_sigmoid), 
                y=post_label(masks),
            )
            iou_metric(
                y_pred=post_pred(outputs_sigmoid),
                y=post_label(masks),
            )

    mean_dice = dice_metric.aggregate().item()
    mean_iou = iou_metric.aggregate().item()
    dice_metric.reset()
    iou_metric.reset()

    return epoch_loss / len(dataloader), mean_dice, mean_iou

# Funzione principale per l'esecuzione
def main():
    # Definizione dei parametri
    data_root = './data/lgg-mari-segmentation' # Da modificare
    print(f"Data root: {data_root}")
    image_folders = glob.glob(os.path.join(data_root, 'TCGA*'))
    print(f"Image folders found: {image_folders}")
    for folder in image_folders:
        patient_id = os.path.basename(folder)
        img_files = glob.glob(os.path.join(folder, f'TCGA*_{patient_id}_*.tif'))
        mask_files = glob.glob(os.path.join(folder, f'TCGA*_{patient_id}_*_mask.tif'))
        print(f"Folder: {folder}, Image files: {img_files}, Mask files: {mask_files}")
        image_paths.extend(img_files)
        mask_paths.extend(mask_files)
    train_ratio = 0.8
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preparazione dei dati
    image_paths = []
    mask_paths = []
    for folder in image_folders:
        patient_id = os.path.basename(folder)
        img_files = glob.glob(os.path.join(folder, f'TCGA*_{patient_id}_*.tif'))
        mask_files = glob.glob(os.path.join(folder, f'TCGA*_{patient_id}_*_mask.tif'))
        image_paths.extend(img_files)
        mask_paths.extend(mask_files)

    # Assicurati che ci sia una corrispondenza tra immagini e maschere
    data = pd.DataFrame({
        'image_path': image_paths,
        'mask_path': mask_paths,
    })
    # Potrebbe essere utile estrarre l'ID del paziente e unirlo con data.csv per analisi future

    train_df, val_df = train_test_split(
        data, 
        test_size=1-train_ratio, 
        random_state=42,
    )

    # Definizione delle trasformazioni
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Ridimensionamento per uniformità
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizzazione ImageNet
    ])

    # Creazione dei Dataset e DataLoader
    train_dataset = MRIDataset(img_dir='', mask_dir='', transform=transform)
    val_dataset = MRIDataset(img_dir='', mask_dir='', transform=transform)
    train_dataset.img_paths = train_df['image_path'].tolist()
    train_dataset.mask_paths = train_df['mask_path'].tolist()
    val_dataset.img_paths = val_df['image_path'].tolist()
    val_dataset.mask_paths = val_df['mask_path'].tolist()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Inizializzazione dei modelli da segmentation_models_pytorch
    unet_model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1,
    ).to(device)
    attention_unet_model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1, 
        decoder_attention_type='scse',
    ).to(device) # Esempio di Attention U-Net
    unet_plusplus_model = smp.UnetPlusPlus(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1,
    ).to(device)

    # Definizione della funzione di loss e degli ottimizzatori
    criterion = nn.BCEWithLogitsLoss() # Binary Cross-Entropy with Sigmoid
    optimizer_unet = optim.Adam(unet_model.parameters(), lr=learning_rate)
    optimizer_att_unet = optim.Adam(attention_unet_model.parameters(), lr=learning_rate)
    optimizer_unet_plusplus = optim.Adam(unet_plusplus_model.parameters(), lr=learning_rate)

    # Definizione delle metriche MONAI
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    post_pred = AsDiscrete(threshold=0.5)
    post_label = AsDiscrete(to_onehot=1)

    # Ciclo di training e valutazione per U-Net
    print("Training U-Net...")
    for epoch in range(num_epochs):
        train_loss = train_model(
            unet_model, 
            train_loader, 
            criterion, 
            optimizer_unet, 
            device,
        )
        val_loss, val_dice, val_iou = evaluate_model(
            unet_model, 
            val_loader, 
            criterion, 
            device, 
            dice_metric, 
            iou_metric, 
            post_pred, 
            post_label,
        )
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

    # Ciclo di training e valutazione per Attention U-Net
    print("\nTraining Attention U-Net...")
    for epoch in range(num_epochs):
        train_loss = train_model(
            attention_unet_model, 
            train_loader, 
            criterion, 
            optimizer_att_unet, 
            device,
        )
        val_loss, val_dice, val_iou = evaluate_model(
            attention_unet_model, 
            val_loader, 
            criterion, 
            device, 
            dice_metric, 
            iou_metric, 
            post_pred, 
            post_label,
        )
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

    # Ciclo di training e valutazione per U-Net++
    print("\nTraining U-Net++...")
    for epoch in range(num_epochs):
        train_loss = train_model(
            unet_plusplus_model, 
            train_loader, 
            criterion, 
            optimizer_unet_plusplus, 
            device, 
            dice_metric, 
            iou_metric, 
            post_pred, 
            post_label,
        )
        val_loss, val_dice, val_iou = evaluate_model(
            unet_plusplus_model, 
            val_loader, 
            criterion, 
            device, 
            dice_metric, 
            iou_metric, 
            post_pred, 
            post_label,
        )
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

if __name__ == '__main__':
    main()