'''
Script per il pre-processing delle immagini e delle maschere.
Per ogni paziente viene estratto solo il canale FLAIR (indice 1), normalizzazione
tra 0 e 1, ridimensionamento a 256x256, maschere binarizzate (0/1).
Le immagini non vengono ruotate essendo già allineate correttamente (non viene
applicata nessuna data augmentation).
Come iutput vengono generati degli array NumPy con shape (1, 256, 256) compatibili
con PyTorch/MONAI.
'''
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configurazioni
RAW_DATA_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
TARGET_SIZE = (256, 256)

def process_patient(patient_dir):
    '''
        Elabora tutte le immagini di un singolo paziente
    '''

    patient_path = os.path.join(RAW_DATA_DIR, patient_dir)
    output_path = os.path.join(PROCESSED_DIR, patient_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Trova tutte le immagini (escludendo le maschere)
    all_files = os.listdir(patient_path)
    image_files = [
        f for f in all_files if f.endswith('.tif') and '_mask' not in f
    ]
    
    for img_file in image_files:
        base_name = img_file.split('.')[0]
        mask_file = f"{base_name}_mask.tif"
        
        # Path completi
        img_path = os.path.join(patient_path, img_file)
        mask_path = os.path.join(patient_path, mask_file)
        
        # Caricamento immagini
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Estrai solo canale FLAIR (secondo canale)
        flair_img = img[:, :, 1]  # Assumendo ordine: pre-contrast, FLAIR, post-contrast
        
        # Preprocessing
        img_preprocessed = preprocess_image(flair_img)
        mask_preprocessed = preprocess_mask(mask)
        
        # Salvataggio
        np.save(os.path.join(output_path, f"{base_name}_img.npy"), img_preprocessed)
        np.save(os.path.join(output_path, f"{base_name}_mask.npy"), mask_preprocessed)

def preprocess_image(img):
    '''
        Preprocessing dell'immagine MRI
    '''

    # Normalizzazione [0, 1]
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Ridimensionamento
    img_resized = cv2.resize(img_norm, (TARGET_SIZE[1], TARGET_SIZE[0]))
    
    # Aggiungi dimensione del canale
    return np.expand_dims(img_resized, axis=0)  # Shape: (1, H, W)

def preprocess_mask(mask):
    '''
        Preprocessing della maschera
    '''

    # Binarizzazione
    _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    # Ridimensionamento
    mask_resized = cv2.resize(
        mask_bin, 
        (TARGET_SIZE[1], TARGET_SIZE[0]), 
        interpolation=cv2.INTER_NEAREST,
    )
    
    return np.expand_dims(mask_resized, axis=0)  # Shape: (1, H, W)

def visualize_sample(patient_dir):
    '''
        Visualizza un campione a caso
    '''

    sample_path = os.path.join(PROCESSED_DIR, patient_dir)
    sample_file = [f for f in os.listdir(sample_path) if '_img.npy' in f][0]
    
    img = np.load(os.path.join(sample_path, sample_file))
    mask = np.load(os.path.join(sample_path, sample_file.replace('_img', '_mask')))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(img[0], cmap='gray')
    plt.title('FLAIR preprocessato')
    
    plt.subplot(132)
    plt.imshow(mask[0], cmap='gray')
    plt.title('Maschera')
    
    plt.subplot(133)
    plt.imshow(img[0], cmap='gray')
    plt.imshow(mask[0], alpha=0.3, cmap='jet')
    plt.title('Overlay')
    
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    # Crea directory di output
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Elabora tutti i pazienti
    patient_dirs = [d for d in os.listdir(RAW_DATA_DIR) if d.startswith('TCGA')]
    
    for patient_dir in tqdm(patient_dirs, desc='Processing patients'):
        process_patient(patient_dir)
    
    # Visualizza un campione
    visualize_sample(patient_dirs[0])