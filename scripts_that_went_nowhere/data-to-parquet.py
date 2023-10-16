##a function to store the data in parquet file
##maybe to give this approach to finetuning a whirl:
##https://dienhoa.github.io/dhblog/posts/finetune_clip.html
import pandas as pd
import cv2
import pyarrow.parquet as pq
import pyarrow as pa
from urllib.request import urlopen
import numpy as np

batch_size = 500

def load_image(row):
    try:
        url = row['media__uri']
        extension = url.split('.')[-1]
        media_uuid = url.split('/')[-1]
        file_name = f'{media_uuid}.{extension}'
       
        with urlopen(url) as u:
            s = u.read()
        arr = np.asarray(bytearray(s), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        return img.tobytes()
    except:
        return None

df_parquet = pd.DataFrame()
reader = pd.read_csv('csv_data/artifact_images_w_descriptions.csv', chunksize=batch_size)

chunks = []  # List to keep batches

for chunk in reader:
    chunk['image'] = chunk.apply(load_image, axis=1)
    chunk['captions'] = chunk[['item__earliest', 'item__latest', 'context___1', 'context___2', 'context___3',
                               'Consists of (Label) [https://erlangen-crm.org/current/P45_consists_of]', 
                               'project_specific_descriptions']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    chunks.append(chunk)

# Concatenate all chunks into a single DataFrame
df_parquet = pd.concat(chunks, ignore_index=True)

table = pa.Table.from_pandas(df_parquet)
pq.write_table(table, 'data.parquet')
