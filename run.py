import os
import glob
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import json
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Dataset personalizzato per le immagini MRI e le maschere
class MRIDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_paths = []
        self.mask_paths = []
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Carica immagine e maschera
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # Converti in scala di grigi (L)

        # Normalizza la maschera a 0 e 1 e aggiungi una dimensione per il canale
        mask = np.array(mask) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0) # [1, H, W]

        if self.transform:
            image = self.transform(image)
            # Ridimensiona la maschera alla stessa dimensione dell'immagine, usando interpolazione NEAREST
            mask = transforms.functional.resize(
                mask, 
                image.shape[1:], 
                interpolation=transforms.functional.InterpolationMode.NEAREST,
            )

        return image, mask

# Funzione per l'addestramento del modello
def train_model(
    model, 
    dataloader, 
    criterion, 
    optimizer, 
    device
):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks) # BCEWithLogitsLoss gestisce i logits
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Funzione per la valutazione del modello con calcolo manuale di Dice e IoU
def evaluate_model(
    model, 
    dataloader, 
    criterion, 
    device
): 
    model.eval()
    epoch_loss = 0
    total_dice = 0
    total_iou = 0
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device) # Maschera originale [B, 1, H, W] con valori 0.0 o 1.0
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()

            # Previsioni binarie: applica sigmoid e soglia a 0.5
            preds_binary = (torch.sigmoid(outputs) > 0.5).float() # [B, 1, H, W] con valori 0.0 o 1.0

            # Calcolo del Dice Score e IoU per ogni batch
            intersection = (preds_binary * masks).sum()
            union = (preds_binary + masks).sum() - intersection 

            # Dice Score (per l'intera batch)
            epsilon = 1e-6 
            dice_score_batch = (2. * intersection + epsilon) / (preds_binary.sum() + masks.sum() + epsilon)
            total_dice += dice_score_batch.item()

            # IoU (per l'intera batch)
            iou_batch = (intersection + epsilon) / (union + epsilon)
            total_iou += iou_batch.item()
            
            num_batches += 1

    mean_dice = total_dice / num_batches
    mean_iou = total_iou / num_batches

    return epoch_loss / len(dataloader), mean_dice, mean_iou

def main():
    # Parametri di configurazione
    data_root = './data/lgg-mri-segmentation'
    train_ratio = 0.8
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Data root: {data_root}")

    # Raccolta dei percorsi delle immagini e delle maschere
    image_paths = []
    mask_paths = []
    image_folders = glob.glob(os.path.join(data_root, 'TCGA*'))
    print(f"Image folders found: {len(image_folders)}")

    for folder in image_folders:
        all_tif_files = glob.glob(os.path.join(folder, '*.tif'))
        current_images = sorted([f for f in all_tif_files if '_mask.tif' not in f])
        current_masks = sorted([f for f in all_tif_files if '_mask.tif' in f])

        for img_path in current_images:
            base_name = os.path.basename(img_path)
            mask_base_name = base_name.replace('.tif', '_mask.tif')
            expected_mask_path = os.path.join(folder, mask_base_name)

            if expected_mask_path in current_masks:
                image_paths.append(img_path)
                mask_paths.append(expected_mask_path)

    if len(image_paths) != len(mask_paths):
        print("Error: Mismatch in number of images and masks collected!")
        sys.exit(1)
    else:
        print(f"\nSuccessfully collected {len(image_paths)} image-mask pairs.")

    print("\nPerforming a small data exploration for class imbalance...")
    sample_size = int(len(mask_paths) * 0.20) # Analizza il 20% delle maschere
    print(f"Analyzing {sample_size} random masks for class imbalance...") # Aggiungi un messaggio più chiaro    
    random_indices = np.random.choice(len(mask_paths), size=sample_size, replace=False)
    
    total_tumor_pixels = 0
    total_pixels = 0

    for i in random_indices:
        mask = Image.open(mask_paths[i]).convert('L')
        mask_array = np.array(mask)
        total_tumor_pixels += np.sum(mask_array > 0) # Assumendo che il tumore sia rappresentato da pixel > 0
        total_pixels += mask_array.size

    if total_pixels > 0:
        tumor_percentage = (total_tumor_pixels / total_pixels) * 100
        print(f"Average tumor pixel percentage across {sample_size} random masks: {tumor_percentage:.2f}%")
        if tumor_percentage < 5: # Un valore soglia per indicare un forte squilibrio
            print("Note: This dataset shows significant class imbalance (tumor pixels are a small minority).")
            print("      This is common in medical imaging and loss functions like BCEWithLogitsLoss (which you are using) are robust to it,")
            print("      but more advanced techniques like Dice Loss or Focal Loss could be considered for future improvements.")
    else:
        print("Could not calculate tumor percentage: No pixels found in sample masks.")
    
    # Creazione del DataFrame e split train/validation
    data = pd.DataFrame({'image_path': image_paths, 'mask_path': mask_paths})
    print(f"DataFrame head:\n{data.head()}")
    print(f"Total samples in DataFrame: {len(data)}")

    if not data.empty:
        train_df, val_df = train_test_split(data, test_size=1-train_ratio, random_state=42)
        print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    else:
        print("Error: No data found after processing. Check file paths and naming conventions.")
        sys.exit(1)

    # Definizione delle trasformazioni per le immagini
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Inizializzazione dei dataset e dataloader
    train_dataset = MRIDataset(img_dir='', mask_dir='', transform=transform)
    val_dataset = MRIDataset(img_dir='', mask_dir='', transform=transform)

    train_dataset.img_paths = train_df['image_path'].tolist()
    train_dataset.mask_paths = train_df['mask_path'].tolist()
    val_dataset.img_paths = val_df['image_path'].tolist()
    val_dataset.mask_paths = val_df['mask_path'].tolist()

    # I dataloader utilizzeranno il nuovo batch_size ridotto
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=os.cpu_count() // 2 or 1)

    # Definizione dei modelli
    models_to_train = {
        "U-Net": smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1, # 1 canale per la segmentazione binaria (foreground)
        ).to(device),
        "Attention U-Net": smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            decoder_attention_type='scse',
        ).to(device),
        "LinkNet": smp.Linknet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        ).to(device)
    }

    # Funzione di loss: Binary Cross-Entropy con Logits
    criterion = nn.BCEWithLogitsLoss()

    # Preparazione per il logging e il salvataggio dei risultati
    training_logs = {}
    output_dir = 'training_results'
    os.makedirs(output_dir, exist_ok=True)

    # Ciclo di addestramento per ogni modello
    for model_name, model in models_to_train.items():
        print(f"\nTraining {model_name}...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        val_losses = []
        val_dices = []
        val_ious = []
        best_val_dice = -1.0

        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_dice, val_iou = evaluate_model(model, val_loader, criterion, device) 

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            val_ious.append(val_iou)

            print(f"Epoch {epoch+1}/{num_epochs} - {model_name}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

            # Salvataggio del modello migliore in base al Dice Score di validazione
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                model_save_path = os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_best_model.pth')
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved best {model_name} model to {model_save_path} with Dice: {best_val_dice:.4f}")

        # Registrazione dei log di training per il modello corrente
        training_logs[model_name] = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_dice': val_dices,
            'val_iou': val_ious
        }

        # Generazione e salvataggio dei grafici delle metriche
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.title(f'{model_name} Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), val_dices, label='Validation Dice Score')
        plt.plot(range(1, num_epochs + 1), val_ious, label='Validation IoU Score')
        plt.title(f'{model_name} Metrics over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_metrics_plot.png'))
        plt.close()
    
    # Salvataggio del riepilogo dei log in formato JSON
    log_json_path = os.path.join(output_dir, 'training_summary.json')
    with open(log_json_path, 'w') as f:
        json.dump(training_logs, f, indent=4)
    print(f"\nTraining summary saved to {log_json_path}")

    # Salvataggio dei log dettagliati in formato CSV per ogni modello
    for model_name, logs in training_logs.items():
        df_logs = pd.DataFrame(logs)
        df_logs.index.name = 'Epoch'
        df_logs.index = df_logs.index + 1 
        csv_path = os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}_training_log.csv')
        df_logs.to_csv(csv_path)
        print(f"Training log for {model_name} saved to {csv_path}")


if __name__ == '__main__':
    main()