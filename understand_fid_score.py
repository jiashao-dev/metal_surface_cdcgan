import torch
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchvision import datasets
import torchvision.transforms as transforms

fid = FID(192)

dataset = datasets.ImageFolder(
    "./fid",
    transform=transforms.Compose(
        [
            transforms.Resize(299),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ]
    )
)

same = (dataset[2][0] * 255).to(dtype=torch.uint8)
diff = (dataset[0][0] * 255).to(dtype=torch.uint8)
good = (dataset[1][0] * 255).to(dtype=torch.uint8)

same = same.unsqueeze(0).repeat(2, 1, 1, 1)
diff = diff.unsqueeze(0).repeat(2, 1, 1, 1)
good = good.unsqueeze(0).repeat(2, 1, 1, 1)

fid.update(same, real=True)
fid.update(same, real=False)
score = fid.compute()

print("FID Score between two same images: %.4f\n\n" % score)

fid.reset()

fid.update(same, True)
fid.update(diff, False)
score = fid.compute()

print("FID Score between two completely different images: %.4f\n\n" % score)

fid.reset()

fid.update(same, True)
fid.update(good, False)
score = fid.compute()

print("FID Score between two identical images: %.4f\n\n" % score)