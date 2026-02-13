import sys
import csv
import logging
from typing import Optional, Union, Any, Dict, Callable
from pathlib import Path, PurePath

sys.path.append(str(PurePath(__file__).parents[1]))  # noqa: E402
from lib.html_embedder import HtmlEmbedder
from lib.file_source import HtmlFile, CsvFile, JsonFile, FileSource
from lib.custom.converter import md_converter_sg_acts
from lib.custom.spliter import md_split_sg_acts
from config import embed_in_use, lc_cll_in_use
from lib.logger import logger


class SGActEmbedder(HtmlEmbedder):
    '''
    HTML Embedder specific for SG Law Acts with
    - Override converter for html to markdown
    - Override spliter to chunk markdown files
    '''

    def __init__(self, name: str,
                 original_file: FileSource,
                 ):

        super().__init__(
            name,
            original_file,

            md_converter=md_converter_sg_acts,
            md_splitter=md_split_sg_acts,
            embed_func=embed_in_use,
        )
        return

    def convert_html_to_md(self):
        return super().convert_html_to_md(save_to_folder='./data/acts_md')

    def chunk_md(self):
        return super().chunk_md(save_to_folder='./data/acts_chunked')

    def embed(self):
        return super().embed(save_to_folder=f"./data/acts_embedding/{embed_in_use.model}")


def embed_sg_acts():
    # Get list of SG laws from csv
    with open('./data/acts.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        acts = [row[0] for row in reader]

    acts = [f'{a} - Singapore Statutes Online' for a in acts]

    # Process and embed each act one by one
    for act in acts:
        p = SGActEmbedder(
            name=act,
            original_file=HtmlFile(
                html_dir=f'./data/acts_html/{act}.html', url=""),
        )

        p.md_file = FileSource(
            file_dir=f'./data/acts_md/{act}.md', format='md')

        p.chunk_json = JsonFile(
            json_dir=f'./data/acts_chunked/{act}.json')

        p.embed_json = JsonFile(
            json_dir=f'./data/acts_embedding/{embed_in_use.model}/{act}.json')

        logger.info(f"Convert Html to Md: {act}")
        # p.convert_html_to_md()  # If already done, may comment off to skip

        logger.info(f"Chunk Md: {act}")
        # p.chunk_md()  # If already done, may comment off to skip

        logger.info(f'Embedding: {act}')
        # p.embed()  # If already done, may comment off to skip

    return


def add_embed_to_vector_store(embedding_data_folder: Union[str, Path], lc_cll: any):
    for data in list(Path(embedding_data_folder).glob('*.json')):
        logger.info(f'Add to vector store: {data}')
        embed_json = JsonFile(json_dir=data).read()

        lc_cll._collection.add(
            documents=[d['page_content'] for d in embed_json],
            embeddings=[d['embedding'] for d in embed_json],
            metadatas=[d['metadata'] for d in embed_json],
            ids=[d['id'] for d in embed_json]
        )

    return


if __name__ == "__main__":
    '''
    Pls comment off below steps based on need
    '''

    # To embed sg acts.
    # embed_sg_acts()

    # To delete existing Chroma db collection.
    lc_cll_in_use.delete_collection()

    # To store embedding vector to vector store.
    add_embed_to_vector_store(
        f'./data/acts_embedding/{embed_in_use.model}', lc_cll_in_use)
    pass
