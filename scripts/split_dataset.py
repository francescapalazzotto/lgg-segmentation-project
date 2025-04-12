'''
RESULTS:
Pazienti totali nel CSV: 110
Pazienti totali nel CSV: 110
Pazienti trovati in processed: 110
Esempio paziente valido: TCGA_CS_4941_19960909 -> death01=1.0

Split completato con successo!
Train: 87 pazienti
Validation: 11 pazienti
Test: 11 pazienti
'''

import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

def create_splits(
    test_size=0.1,
    val_size=0.1, 
    seed=42
):
    '''
        Crea split train/val/test preservando la distribuzione:
            - Stratificato per 'death01' (outcome)
            - Per paziente (non per slice)
    '''
    # Path corretti (relativi alla posizione dello script)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_path = os.path.join(data_dir, 'data.csv')
    processed_dir = os.path.join(data_dir, 'processed')
    splits_dir = os.path.join(data_dir, 'splits')
    
    # Debug: verifica esistenza cartelle
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Cartella processed non trovata: {processed_dir}")
    
    # Caricamento dati
    try:
        df = pd.read_csv(csv_path)
        patients = [p for p in os.listdir(processed_dir) if p.startswith('TCGA')]
    except Exception as e:
        raise RuntimeError(f"Errore caricamento dati: {str(e)}")

    if len(patients) == 0:
        raise ValueError("Nessun paziente trovato nella cartella processed")

    # Estrai ID base (TCGA_XX_XXXX senza suffissi)
    base_patients = ['_'.join(p.split('_')[:3]) for p in patients]
    
    # Filtra pazienti validi
    valid_patients = []
    valid_death01 = []
    
    for patient_csv in df['Patient']:
        if patient_csv in base_patients:
            idx = base_patients.index(patient_csv)
            original_name = patients[idx]  # Conserva nome originale con suffisso
            
            death_val = df.loc[df['Patient'] == patient_csv, 'death01'].values[0]
            if not pd.isna(death_val):
                valid_patients.append(original_name)  # Usa il nome della cartella
                valid_death01.append(death_val)

    if len(valid_patients) == 0:
        raise ValueError(
            "Nessun paziente valido trovato. Controlla:\n"
            f"Esempio nome CSV: {df['Patient'].iloc[0]}\n"
            f"Esempio nome cartella: {patients[0] if len(patients) > 0 else 'N/A'}\n"
            f"Valori NaN in 'death01': {df['death01'].isna().sum()}/{len(df)}"
        )

    # Debug info
    print(f"Pazienti totali nel CSV: {len(df)}")
    print(f"Pazienti trovati in processed: {len(patients)}")
    print(f"Pazienti validi per split: {len(valid_patients)}")
    print(f"Esempio paziente valido: {valid_patients[0]} -> death01={valid_death01[0]}")

    # Split
    try:
        train_val, test = train_test_split(
            valid_patients,
            test_size=test_size,
            stratify=valid_death01,
            random_state=seed
        )
        
        # Prepara i dati per il secondo split
        train_val_death = [valid_death01[valid_patients.index(p)] for p in train_val]
        
        train, val = train_test_split(
            train_val,
            test_size=val_size/(1-test_size),
            stratify=train_val_death,
            random_state=seed
        )
    except Exception as e:
        raise RuntimeError(f"Errore durante lo split: {str(e)}")

    # Crea cartella splits se non esiste
    os.makedirs(splits_dir, exist_ok=True)
    
    # Salva gli split
    splits = {
        'train': train,
        'val': val,
        'test': test,
    }
    
    splits_path = os.path.join(splits_dir, 'splits.json')
    with open(splits_path, 'w') as f:
        json.dump(splits, f)
    
    print("\nSplit completato con successo!")
    print(f"Train: {len(train)} pazienti")
    print(f"Validation: {len(val)} pazienti")
    print(f"Test: {len(test)} pazienti")
    print(f"File salvato in: {splits_path}")

if __name__ == '__main__':
    create_splits()