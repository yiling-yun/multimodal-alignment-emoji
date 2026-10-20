import pandas as pd
import pickle

if __name__ == '__main__':
    full_emoji = pd.read_csv('data/emoji/full_emoji.csv')

    with open('./dc_img/train_embeddings.pkl', 'rb') as handle:
        train_df = pickle.load(handle)

    with open('./dc_img/test_embeddings.pkl', 'rb') as handle:
        test_df = pickle.load(handle)

    save_data = {'name': [], 'company': [], 'embedding': []}

    for name in full_emoji['name']:
        if name in train_df:
            embeddings = train_df[name]
        else:
            embeddings = test_df[name]

        for company in embeddings:
            save_data['name'].append(name)
            save_data['company'].append(company)
            save_data['embedding'].append(' '.join(embeddings[company]['embedding'].astype(str)))

    save_data = pd.DataFrame(save_data)
    save_data.to_csv('dc_img/emoji_embeddings.csv', index=False)