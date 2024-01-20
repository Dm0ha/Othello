import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

class ResidualModel(nn.Module):
    """
    The model for a residual block that contains an S.E. block
    """
    def __init__(self, filters, se_reduction=16):
        super().__init__()
        self.c1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.b1 = nn.BatchNorm2d(filters)
        self.c2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.b2 = nn.BatchNorm2d(filters)
        # S.E.
        latent_size = filters // se_reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(filters, latent_size)
        self.l2 = nn.Linear(latent_size, filters)

    def forward(self, x):
        original = x
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        # S.E.
        s = self.pool(x).view(x.size()[0], x.size()[1]) # remove the 1 by 1 dimension
        s = self.l2(F.relu(self.l1(s)))
        s = F.sigmoid(s).view(x.size()[0], x.size()[1], 1, 1) # add the 1 by 1 dimension back
        x = x * s.expand(x.size())
        # Residual connection
        x = x + original
        return F.relu(x)

class OthelloModel(nn.Module):
    """
    The overall Othello model
    """
    def __init__(self, filters=128):
        super().__init__()
        input_layers = 5
        self.c1 = nn.Conv2d(input_layers, filters, 3, padding=1)
        self.res = nn.Sequential(*[ResidualModel(filters) for _ in range(10)])
        self.l1 = nn.Linear(128 * 8 * 8, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.res(x)
        x = x.view(x.size()[0], -1) # flatten to channels * height * width
        x = F.relu(self.l1(x))
        x = F.sigmoid(self.l2(x))
        return x

class GameDataset(Dataset):
    """
    Class for a game dataset
    """
    def __init__(self, x, y):
        """
        Args:
            x (np.array): input games
            y (np.array): winner
        """
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        
    def __getitem__(self, ix):
        """Get a sample from the dataset
        Args:
            ix (int): index
        Returns:
            tuple: (input, winner)
        """
        return self.x[ix], self.y[ix]
    
    def __len__(self):
        """Get the length of the dataset
        Returns:
            int: length of the dataset
        """
        return len(self.x)

class ModelGenerator:
    @staticmethod    
    def generate_model(epochs, data_file, out_folder, base_model_path=None):
        """
        Train an Othello model.
        Args:
            epochs (int): number of epochs to train for
            data_file (str): path to the data file
            out_folder (str): folder path to save model
            base_model_path (str): path to the model with base weights
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        data = np.load(data_file)
        x = data[:, :-1, :, :]
        y = data[:, -1, 0, 0]

        # Shuffle
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        x_train = x[:int(0.8 * len(x))]
        y_train = y[:int(0.8 * len(y))]
        x_test = x[int(0.8 * len(x)):]
        y_test = y[int(0.8 * len(y)):]
        train_dataset = GameDataset(x_train, y_train)
        test_dataset = GameDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = OthelloModel().to(device)
        if base_model_path is not None: # start with weights from a different model
            model.load_state_dict(torch.load(base_model_path))
        loss_fn = nn.MSELoss()
        opt = Adam(model.parameters(), lr=0.0001)
        lowest_test = [999, 0]

        for epoch in range(epochs):
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for seen, batch in enumerate(train_loader):
                x, y = batch[0].to(device), batch[1].to(device)
                model.train()
                opt.zero_grad()
                loss = loss_fn(model(x).view(-1), y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss/(seen + 1))
                progress_bar.update()
            progress_bar.close()
            torch.save(model.state_dict(), f"{out_folder}/model_{epoch+1}.pth")

            test_loss = 0.0
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    model.eval()
                    out = model(x).view(-1)
                    preds.append(out.cpu().numpy())
                    loss = loss_fn(out, y)
                    test_loss += loss.item()

            print(f"Test Loss: {test_loss/len(test_loader)}")
            preds = np.concatenate(preds)
            print(f"Average Prediction: {np.mean(preds)}")
            if test_loss/len(test_loader) < lowest_test[0]:
                lowest_test[0] = test_loss/len(test_loader)
                lowest_test[1] = epoch + 1
            print(f"Lowest Test Loss: epoch {lowest_test[1]}")

if __name__ == "__main__":
    ModelGenerator.generate_model(200, "self_games.npy", "models_hm2", "models_hm/model_18.pth")