import numpy as np
import pandas as pd
import pickle
import faiss

from sklearn.preprocessing import normalize

_VECTOR_COLUMNS = list(map(str, range(100)))
__all__ = (
    "load_rec_model",
    "get_similar_songs",
    "recommend_songs",
    "reduce_memory_usage",
    "get_top_songs_by_artists"
)


def load_rec_model(model_path: str):
    with open(model_path, 'rb') as file:
        return pickle.load(file)


def get_similar_songs(vector: np.ndarray, dataset: pd.DataFrame, k: int = 5):
    item_embeddings = normalize(dataset[_VECTOR_COLUMNS], axis=1)
    test_vector = normalize(vector.reshape(1, -1), axis=1)

    dimension = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(item_embeddings)

    _, indices = index.search(test_vector, k)

    return dataset.loc[indices[0], ["id", "artists", "name"]].to_dict()


def recommend_songs(ids: list[str], dataset: pd.DataFrame, model, k: int = 5):
    query_indices = dataset[dataset["id"].isin(ids)].index
    embeddings = dataset.loc[:, '0':'99'].to_numpy()
    queries = embeddings[query_indices]

    distances, indices = model.kneighbors(queries)

    rec_indexes = [
        neighbor_idx
        for i, query_idx in enumerate(query_indices)
        for neighbor_idx in indices[i][1:]
    ]
    rec_songs = dataset.loc[
        rec_indexes, ["id", "artists", "name"]
    ].drop_duplicates(subset="name")
    rec_songs = rec_songs[~rec_songs["id"].isin(ids)]
    rec_songs = rec_songs.sample(k * len(ids)).reset_index(drop=True)
    return rec_songs.to_dict(orient="records")


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype.name
        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

