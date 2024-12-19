import os
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('punkt')

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }
        self.stoi = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        self.freq_threshold = freq_threshold

    def tokenize(self, text):
        return word_tokenize(text.lower())

    def make_vocabulary(self, sequences):
        current_idx = 4
        frequencies = {}

        for sequence in sequences:
            for word in self.tokenize(sequence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = current_idx
                    self.itos[current_idx] = word
                    current_idx += 1

    def encode(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

    def decode(self, sequence):
        return [self.itos[token] if token in self.itos else "<UNK>" for token in sequence]

    def __len__(self):
        return len(self.itos)

class CustomPKMNDataset(Dataset):
    def __init__(self, img_dir, data_file, transform=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(data_file)
        self.transform = transform
        self.vocab = Vocabulary()
        self.vocab.make_vocabulary(self.data['caption'])
        
        # Create a mapping for labels (types) to integers
        self.label_mapping = {label: idx for idx, label in enumerate(self.data['type'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)  # This returns a tensor
        image = TF.to_pil_image(image)  # Convert to PIL Image
        caption = self.data.iloc[idx, 2]
        caption_encoded = self.vocab.encode(caption)

        label = self.data.iloc[idx, 1]  # Label (type) as string
        label_idx = self.label_mapping[label]  # Convert label to integer index

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(caption_encoded), label_idx

class PaddingCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        types = [item[2] for item in batch]
        types = torch.tensor(types)
        return imgs, targets, types

def make_loader(img_dir, data_file, transform, batch_size=32, num_workers=0, shuffle=True, pin_memory=True):
    dataset = CustomPKMNDataset(img_dir, data_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
        pin_memory=pin_memory, collate_fn=PaddingCollate(pad_idx))
    return dataloader, dataset
