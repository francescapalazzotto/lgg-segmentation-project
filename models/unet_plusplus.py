import segmentation_models_pytorch as smp
import torch

def get_unet_plusplus_model():
    return smp.UnetPlusPlus(
        encoder_name='resnet34',      # Encoder preaddestrato
        encoder_weights='imagenet',   # Transfer learning
        in_channels=1,                # Input a 1 canale (FLAIR)
        classes=1,                    # Output a 1 canale
        activation='sigmoid',         # Output binario [0,1]
        encoder_depth=4,              # Usa 4 blocchi encoder
        decoder_channels=[256, 128, 64, 32],  # Canali decoder
        decoder_use_batchnorm=True,   # BatchNorm per stabilità
        decoder_attention_type=None   # Disabilita attention se non serve
    )

# Esempio d'uso
# model = get_unet_plusplus_model()
# x = torch.rand(2, 1, 256, 256)
# print(model(x).shape)  # Output: torch.Size([2, 1, 256, 256])