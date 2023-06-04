"""
Download it from here:
https://dumps.wikimedia.org/backup-index.html


"""

import os

import requests
from gensim.corpora import WikiCorpus
from tqdm import tqdm


def download_file(url, filename):
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="B", unit_scale=True, desc=filename)

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                t.update(len(chunk))
                f.write(chunk)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")


url = "https://dumps.wikimedia.your.org/enwiki/20220820/enwiki-20220820-pages-articles-multistream.xml.bz2"
filename = "enwiki-20220820-pages-articles-multistream.xml.bz2"
download_file(url, filename)


def extract_text_from_wikipedia_dump(dump_file_path, output_file_path):
    """
    Extract and save plain text from a Wikipedia dump.

    :param dump_file_path: path to the Wikipedia dump file (*.bz2 file)
    :param output_file_path: path to the output text file

    To run: extract_text_from_wikipedia_dump("enwiki-latest-pages-articles.xml.bz2", "wikipedia_plaintext.txt")

    """
    # Create a WikiCorpus object
    wiki = WikiCorpus(dump_file_path, dictionary={})

    # Open a file for writing extracted text
    with open(output_file_path, "w") as output_file:
        # Iterate over the texts from the dump file and write to the output file
        for i, text in enumerate(wiki.get_texts()):
            # Convert list of tokens to a single string and write to the output file
            output_file.write(" ".join(text) + "\n")

            # Optional: print progress every 10,000 articles
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} articles")

    print("Processing complete!")
