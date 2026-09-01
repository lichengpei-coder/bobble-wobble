# Creating a model using pre-trained EfficientNet
import torch.nn as nn
import timm

class AIImageDetector(nn.Module):
    def __init__(self, num_classes=53):
        super(AIImageDetector, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_output_size = 1280
        # Making a classifier
        self.classifier = nn.Linear(enet_output_size, num_classes)

    def forward(self, x):
        # Connect these parts and return output
        x = self.features(x)
        output = self.classifier(x)
        return output
