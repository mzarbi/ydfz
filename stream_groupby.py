import itertools
import multiprocessing as mp
import pandas as pd
import os


def stream_groupby_csv(path, key, output_dir, chunk_size=1e6, pool=None, **kwargs):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Make sure path is a list
    if not isinstance(path, list):
        path = [path]

    # Chain the chunks
    kwargs['chunksize'] = chunk_size
    chunks = itertools.chain(*[
        pd.read_csv(p, **kwargs)
        for p in path
    ])

    results = []
    orphans = pd.DataFrame()

    for chunk in chunks:
        # Add the previous orphans to the chunk
        chunk = pd.concat((orphans, chunk))

        # Determine which rows are orphans
        last_val = chunk[key].iloc[-1]
        is_orphan = chunk[key] == last_val

        # Put the new orphans aside
        chunk, orphans = chunk[~is_orphan], chunk[is_orphan]

        # If a pool is provided then we use apply_async
        if pool:
            results.append(pool.apply_async(process_chunk, args=(chunk, key, output_dir)))
        else:
            process_chunk(chunk, key, output_dir)

    # Don't forget the remaining orphans
    if len(orphans):
        if pool:
            results.append(pool.apply_async(process_chunk, args=(orphans, key, output_dir)))
        else:
            process_chunk(orphans, key, output_dir)

    # If a pool is used then we have to wait for the results
    if pool:
        for r in results:
            r.get()


def process_chunk(chunk, key, output_dir):
    grouped = chunk.groupby(key)
    for group_name, group_data in grouped:
        file_path = os.path.join(output_dir, f"{group_name}.csv")
        if os.path.exists(file_path):
            group_data.to_csv(file_path, mode='a', header=False, index=False)
        else:
            group_data.to_csv(file_path, mode='w', header=True, index=False)


# Example usage
if __name__ == "__main__":
    path = "large_dataset.csv"
    key = "group_column"
    output_dir = "output_groups"
    chunk_size = 1e6

    # If you want to use multiprocessing
    with mp.Pool(processes=4) as pool:
        stream_groupby_csv(path, key, output_dir, chunk_size, pool=pool)
    # If you don't want to use multiprocessing
    # stream_groupby_csv(path, key, output_dir, chunk_size)
