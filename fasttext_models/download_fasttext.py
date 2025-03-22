import fasttext.util
import json
import os
import re

from orgpackage.clusterembedder import get_all_languages, get_downloaded_languages


ls = get_all_languages()
for language in ls:
    #fasttext.util.download_model(language, if_exists='ignore')
    print(language)

dls = get_downloaded_languages()
print(dls)
print(f'{len(dls)}/{len(ls)}')
