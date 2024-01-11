import cv2 
import numpy as np
import torch
from torch import nn
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),  # Change 1 to 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU()
        )
        #self.fusion = nn.Conv2d(256, 256, 1, padding=0)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid(),
        )
        #self.customized = MyLayer()
        #[0,1] = [0,256] - 128

    def forward(self, x):
        # x = gets the input image
        encoder_features = self.encoder(x)
        x = self.decoder(encoder_features)
        return x





# setting the model out
model = MyModel()
pretrained_model_path = '/Users/bekhzodravshanov/Desktop/PC/IT_CS/Colorize/final_pretrained_model.pth'

if torch.cuda.is_available():
    checkpoint = torch.load(pretrained_model_path)
else:
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))

model.load_state_dict(checkpoint)
model.eval()



video = cv2.VideoCapture(1)
video.set(cv2.CAP_PROP_FPS, 100)



while True:
    # the frameing
    ret , frame = video.read()

    # squaring, mirroring, black-whitening, rescaling 128x128
    frame = np.array(frame)
    frame = np.flip(frame, 1)
    frame = frame[:,int(len(frame[0])//2  - int(len(frame)*0.5)):int(len(frame[0])//2 + int(len(frame)*0.5))]
    frame = cv2.resize(frame, (128,128))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.merge([frame] * 3)
    cv2.imshow('press q to print', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break
frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # Assuming HWC to CHW format conversion
# transfer from [3,128, 128] to [1,3,128,128] 1 is batch here
frame = frame.unsqueeze(0) / 255

# seting to the 
output = model(frame)

output = output.squeeze().detach().numpy()

output = np.transpose(output, (1, 2, 0))
output *= 255
print(output[0])
output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

cv2.imwrite('save1.png', output)
video.release()
# Destroy all the windows 
cv2.destroyAllWindows() 

print("gello")