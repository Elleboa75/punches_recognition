import torch
import torchvision

class ResNetCustom(torchvision.models.ResNet):
    def __init__(self, num_classes:int, architecture:str="resnet18", dim_latent:int=32):
        if architecture == "resnet18":
            layers_list = [2, 2, 2, 2]
            residual_block = torchvision.models.resnet.BasicBlock
        elif architecture == "resnet34":
            layers_list = [3, 4, 6, 3]
            residual_block = torchvision.models.resnet.BasicBlock
        elif architecture == "resnet50":
            layers_list = [3, 4, 6, 3]
            residual_block = torchvision.models.resnet.Bottleneck
        else:
            raise ValueError(f"Unknown architecture {architecture}")
        super().__init__(residual_block, layers_list, num_classes=num_classes)
        del self.fc
        self.fc1 = torch.nn.Linear(512, dim_latent)
        self.fc2 = torch.nn.Linear(dim_latent, num_classes)
    
    def forward(self, x):
        '''
        Forward pass of the model.

        Parameters
        ----------
        x: a four-dimensional torch.Tensor of shape containing the input images.

        Returns
        -------
        out_z: a two-dimensional torch.Tensor of shape (num_datapoints x dim_latent)
        out_y: a two-dimensional torch.Tensor of shape (num_datapoints x num_classes)
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.avgpool(x)
        out = out.reshape(out.shape[0], -1)
        
        out_z = self.fc1(out)
        out_y = self.fc2(out_z)

        return out_z, out_y