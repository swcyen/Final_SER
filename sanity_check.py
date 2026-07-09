from models.encoder import SpeechEncoder
from models.ssl_model import SSLModel
import torch
encoder = SpeechEncoder()

ssl_model = SSLModel(encoder)

wave = torch.randn(4, 1, 80000)

z = ssl_model(wave)

print(z.shape)
