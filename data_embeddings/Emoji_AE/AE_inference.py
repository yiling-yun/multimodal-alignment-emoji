import torch
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Emoji
from autoencoder import to_img, AutoEncoder, hidden_dims, patch_size, model_path, data_dir
import pandas as pd


def inference(model, dataloader):
    model.eval()

    df = {}
    for data in dataloader:
        img, label, companies = data
        assert img.shape[0] == 1
        label = label[0]
        companies = [c[0] for c in companies]
        img = img[0].cuda()
        recon_x, z = model(img)
        img = to_img(img.detach().cpu())
        recon_x = to_img(recon_x.detach().cpu())
        df[label] = {}

        z = z.detach().cpu().numpy()
        for i in range(len(img)):
            df[label][companies[i]] = {'image': img[i], 'recon_image': recon_x[i], 'embedding': z[i]}
    return df


if __name__ == '__main__':
    test = True

    full_emoji = pd.read_csv('data/emoji/full_emoji.csv')

    model = AutoEncoder(3, 300, hidden_dims=hidden_dims).cuda()
    model.load_state_dict(torch.load(model_path))

    t = transforms.Compose([transforms.Resize((patch_size, patch_size)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader = torch.utils.data.DataLoader(
        Emoji(data_dir, transform=t, split='train', inference=True),
        batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        Emoji(data_dir, transform=t, split='test', inference=True),
        batch_size=1, shuffle=False)

    train_df = inference(model, train_loader)
    test_df = inference(model, test_loader)

    with open('./dc_img/train_embeddings.pkl', 'wb') as handle:
        pickle.dump(train_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./dc_img/test_embeddings.pkl', 'wb') as handle:
        pickle.dump(test_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


