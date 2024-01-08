"""
Download it from here:
https://dumps.wikimedia.org/backup-index.html

I stuck a bunch of stuff here. Clean me
"""

import os
import re
import requests
from gensim.corpora import WikiCorpus
from tqdm import tqdm
from gensim import utils
from gensim.corpora import WikiCorpus, wikicorpus
from gensim.corpora.wikicorpus import ARTICLE_MIN_WORDS, IGNORED_NAMESPACES, WikiCorpus, utils

def download_file(url, filename):
    """
    Usage:
    url = "https://dumps.wikimedia.your.org/enwiki/20220820/enwiki-20220820-pages-articles-multistream.xml.bz2"
    filename = "enwiki-20220820-pages-articles-multistream.xml.bz2"
    download_file(url, filename)
    """
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

class PunctuationKeepingWikiCorpus(wikicorpus.WikiCorpus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_pages(self, f):
        elems = (elem for _, elem in utils.iterparse(f, events=("end",)))
        page_tag = "rev" if self.metadata else "page"
        for elem in elems:
            if elem.tag == page_tag and len(elem) > 0:
                text = self.process_article(elem.text, elem.attrib["title"], pagename_to_id=elem.attrib["id"])[0]
                # remove references to files and images before writing to disk
                text = re.sub(r"(File|Image)\:.*\|", "", text)
                text = re.sub(r"\[\[.*\]\]", "", text)
                yield text
            elem.clear()  # not needed any more, free memory

    def process_article(self, text, title, pagename_to_id):
        text = utils.to_unicode(text, "utf8", errors="ignore")
        text = utils.decode_htmlentities(text)  # '&amp;nbsp;' --> '\xa0'
        return text, [], []  # Do not tokenize the text, do not remove stopwords or do any other preprocessing.

    @staticmethod
    def tokenize(text, token_min_len=wikicorpus.TOKEN_MIN_LEN, token_max_len=wikicorpus.TOKEN_MAX_LEN, lower=True):
        # Override gensim.parsing.preprocessing.tokenize here, keeping punctuation
        return [
            token
            for token in utils.tokenize(text, lower=lower, errors="ignore")
            if token_min_len <= len(token) <= token_max_len and not token.startswith("_")
        ]

def extract_text_from_wikipedia_dump(dump_file_path, output_file_path):
    """
    Extract and save plain text from a Wikipedia dump.

    :param dump_file_path: path to the Wikipedia dump file (*.bz2 file)
    :param output_file_path: path to the output text file

    To run: extract_text_from_wikipedia_dump("enwiki-latest-pages-articles.xml.bz2", "wikipedia_plaintext.txt")

    """
    # Create a WikiCorpus object
    # wiki = WikiCorpus(dump_file_path, dictionary={})
    wiki = PunctuationKeepingWikiCorpus(dump_file_path, dictionary={})

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


import bz2

import mwparserfromhell
from gensim.corpora.wikicorpus import extract_pages


def extract_text_from_wikipedia_dump(dump_file_path, output_file_path):
    """
    Extract and save plain text with punctuation from a Wikipedia dump.

    dump_file_path: path to the Wikipedia dump file (*.bz2 file)
    """
    # Open a file for writing extracted text
    with open(output_file_path, "w") as output_file, bz2.open(dump_file_path, mode="rt") as xml_file:
        # Iterate over the texts from the dump file and write to the output file
        for title, text, page_id in extract_pages(xml_file):
            if text[:9] == "#REDIRECT":
                print(f"Skipped redirect {title}")
                continue

            # Parse the raw wikitext using mwparserfromhell
            parsed_text = mwparserfromhell.parse(text)
            # Write the stripped text (removing wiki markup but keeping punctuation) to the output file
            final_text = parsed_text.strip_code()
            output_file.write(final_text + "\n")
            print(f"Processed {title}")

    print("Processing complete!")


extract_text_from_wikipedia_dump("enwiki-20220820-pages-articles-multistream.xml.bz2", "wikipedia_plaintext.txt")
