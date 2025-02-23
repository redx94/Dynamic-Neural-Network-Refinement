
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_set, val_set = random_split(dataset, [50000, 10000])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
