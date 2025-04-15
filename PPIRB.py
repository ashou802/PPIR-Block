class UpsampleWithGuidance(nn.Module):
    def __init__(self, scale, num_feat):
        super(UpsampleWithGuidance, self).__init__()
        self.scale = scale
        self.num_feat = num_feat
        self.num_stages = int(math.log(scale, 2)) if (scale & (scale - 1)) == 0 else 1

        self.reduce_blocks2 = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.pixel_shuffle_blocks = nn.ModuleList()

        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(self.num_stages):
                self.reduce_blocks2.append(nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1))
                self.conv_blocks.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                self.pixel_shuffle_blocks.append(nn.PixelShuffle(2))
        elif scale == 3:
            self.reduce_blocks2.append(nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1))
            self.conv_blocks.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            self.pixel_shuffle_blocks.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')

    def forward(self, x, guidances):
        out = x
        # For each stage of upsampling, combine with the corresponding guidance feature
        for i in range(self.num_stages):
            guidance = guidances[i]
            combined = torch.cat((out, guidance), dim=1)  # Concatenate along the channel dimension
            reduced = self.reduce_blocks2[i](combined)    # Reduce channel dimensions
            out = self.conv_blocks[i](reduced)
            out = self.pixel_shuffle_blocks[i](out)

        return out

class CotGuidanceBlock(nn.Module):
    def __init__(self, num_feat, upscale):
        super(CotGuidanceBlock, self).__init__()

        self.upscale = upscale
        self.num_feat = num_feat

        # Initialize transposed convolution modules, adding appropriate layers for each scale
        self.ConvTransposes = nn.ModuleList()

        if upscale == 2:
            self.ConvTransposes.append(nn.ConvTranspose2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1))
        elif upscale == 3:
            self.ConvTransposes.append(nn.ConvTranspose2d(num_feat, num_feat, kernel_size=4, stride=3, padding=1))
        elif upscale == 4:
            self.ConvTransposes.extend([
                nn.ConvTranspose2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1)
            ])
        elif upscale == 8:
            self.ConvTransposes.extend([
                nn.ConvTranspose2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1)
            ])

        # Initialize downsampling layer
        self.down_sample3 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        guidances = []  # Temporary storage for guidance features in the current forward pass
        # Perform transposed convolution in each step
        for conv in self.ConvTransposes:
            x = conv(x)  # Apply transposed convolution
            downsampled = self.down_sample3(x)  # Apply downsampling after each transposed convolution
            guidances.append(downsampled)

        return guidances  # Return the guidance features from this forward pass
