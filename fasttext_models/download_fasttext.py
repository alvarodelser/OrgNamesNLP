import fasttext.util
import json
import os
import re

from orgpackage.clusterembedder import get_all_languages, get_downloaded_languages
previously_downloaded = set(['mt', 'de', 'el', 'fr', 'lv', 'ro', 'sl'])

ls = get_all_languages()
fasttext.util.download_model('en', if_exists='ignore')
for language in ls:
    if language not in previously_downloaded:
        fasttext.util.download_model(language, if_exists='ignore')

dls = get_downloaded_languages()
print(dls)
print(f'Downloaded {len(dls)}/{len(ls)}')