from torch import nn
from .layers.self_attention import SelfAttentionBlock

class OCDenseNet(nn.Module):
    def __init__(self, num_classes, arch='densenet161'):
        super(OCDenseNet, self).__init__()

        self.densenet = getattr(models, arch)(pretrained=True).features[:7].eval()  # until x8 downscale
        self.conv = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.oc_block = SelfAttentionBlock(512, 512, 512)
        self.classifier = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.upsample = True

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode='replicate')  # cause densenet loose 1 pixel
        # somewhere in downscaling
        x = self.densenet(x)
        x = self.conv(x)
        context = self.oc_block(x)
        x = self.classifier(torch.cat([x, context], dim=1))

        x = F.interpolate(x, scale_factor=8, mode='bilinear')

        return x


def oc_densenet128(num_classes=1, pretrained=False):
    net = OCDenseNet(num_classes)

    if pretrained:
        state_dict = torch.load('./resources')
        net.load_state_dict(state_dict, strict=False)

    return net