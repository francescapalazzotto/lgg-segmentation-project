import segmentation_models_pytorch as smp
import torch

# Configurazione diretta (senza classi)
def get_unet_model():
    return smp.Unet(
        encoder_name='resnet34',  # Encoder preaddestrato su ImageNet
        encoder_weights='imagenet',  # Sfrutta i pesi preaddestrati
        in_channels=1,  # Adattato per input a 1 canale (FLAIR)
        classes=1,  # Output a 1 canale (maschera binaria)
        activation='sigmoid',
        decoder_channels=[256, 128, 64, 32],  # Canali decoder ridotti per bilanciare ResNet
        encoder_depth=4,  # Usa solo i primi 4 blocchi di ResNet (su 5 totali)
        decoder_use_batchnorm=True  # Abilita BN per migliorare la stabilità
    )

# Esempio d'uso:
# model = get_unet_model()
# x = torch.rand(1, 1, 256, 256)  # Input fittizio
# print(model(x).shape)  # Output: torch.Size([1, 1, 256, 256])