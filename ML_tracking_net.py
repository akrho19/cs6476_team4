import torch
import torch.nn as nn


class TrackNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function
        """
        super().__init__()
        print("TrackNet __init__ ")
        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        self.conv_layers.append(nn.Conv3d(1,30,(3,10,10)))
        self.conv_layers.append(nn.Flatten(start_dim=1,end_dim=2))
        self.conv_layers.append(nn.MaxPool2d(3))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(self.make_2D_block(30,40,(7,7)))
        self.conv_layers.append(nn.Dropout(p=0.1))
        self.conv_layers.append(self.make_2D_block(40,50,(5,3)))
        self.conv_layers.append(nn.Dropout(p=0.1))
        self.conv_layers.append(self.make_2D_block(50,60,(5,5)))
        #self.conv_layers.append(nn.Dropout(p=0.1))
        
        self.fc_layers.append(nn.Linear(2100,420))
        self.fc_layers.append(nn.Dropout(p=0.5))
        self.fc_layers.append(nn.Linear(420,84))
        self.fc_layers.append(nn.Dropout(p=0.5))
        self.fc_layers.append(nn.Linear(84,14))

        self.loss_criterion = nn.MSELoss(reduction='mean')

    def make_2D_block(self, in_planes, out_planes, kernel):
        return nn.Sequential(
                nn.Conv2d(in_planes,out_planes,kernel),
                nn.MaxPool2d((3,3)),
                nn.ReLU()
        )

    def forward(self, x: torch.tensor):
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (estimated poses) of the net [Dim: (N,14)]
        """
        conv_features = None  # output of x passed through convolution layers (4D tensor)
        flattened_conv_features = None  # conv_features reshaped into 2D tensor using .reshape()
        model_output = None  # output of flattened_conv_features passed through fully connected layers
        ############################################################################
        # Student code begin
        ############################################################################
        #print(x.shape)
        x = torch.unsqueeze(x,dim=1)
        #print(x.shape)
        conv_features = self.conv_layers.forward(x)
        flat = nn.Flatten()
        flattened_conv_features = flat(conv_features)
        model_output = self.fc_layers.forward(flattened_conv_features)
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
