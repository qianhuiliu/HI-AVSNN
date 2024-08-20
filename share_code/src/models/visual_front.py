from torch import nn
from src.models.layers import *

#visual cue extraction
class Spiking_Visual_front(nn.Module):
    def __init__(self, in_channels=1):
        super(Spiking_Visual_front, self).__init__()
        self.features = nn.Sequential(
            Layer(1,64,3,1,1),
            Layer(64,128,3,2,1),
            Layer(128,256,3,1,1),
            Layer(256,256,3,2,1),
            Layer(256,512,3,1,1),
            Layer(512,512,3,2,1),
            Layer(512,512,3,1,1),
            Layer(512,512,3,2,1),
        )
        W = 3
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,100))
        self.act = LIFSpike()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.dropout(x)
        x = self.classifier(x)
        x = self.act(x)
        return x
