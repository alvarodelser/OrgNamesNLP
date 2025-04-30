import re
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from tqdm import tqdm
import stopwordsiso as stopwords
import os
import pandas as pd
from wikdict_compound import split_compound, make_db

from orgpackage.config import SPACY_MODELS, COUNTRY_DICT


######################################################### STD TOKENIZER ######################################################
def normalize_stopwords(stopwords_list, vectorizer):
    """Applies vectorizer preprocessing (e.g., lowercasing, tokenization) to stopwords"""
    analyzer = vectorizer.build_analyzer()  # Get tokenizer from TfidfVectorizer
    return list(word for sw in stopwords_list for word in analyzer(sw))



#################################################### SPACY TOKENIZER/LEMMATIZER ######################################################
def custom_tokenizer(text, language_code):
    """
    Tokenizes and lemmatizes text based on the specified language using spaCy.
    """
    if language_code in SPACY_MODELS:
        nlp = spacy.load(SPACY_MODELS[language_code], disable=["ner"])
    else:
        return None

    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    if not lemmatized_tokens or all(token.isspace() for token in lemmatized_tokens):
        return None
    return lemmatized_tokens


def tokenize(df, save_path="./results/tokenized_names.csv"):
    # Check if there's a partially saved file
    if os.path.exists(save_path):
        df_saved = pd.read_csv(save_path)
        processed_instances = set(df_saved["instance"])
    else:
        df_saved = pd.DataFrame(columns=["instance", "names", "tokenized"])
        processed_instances = set()

    # Filter rows that haven't been processed yet
    df_to_process = df[~df["instance"].isin(processed_instances)]

    # Process each row
    for i in tqdm(range(len(df_to_process)), desc="Tokenizing Names"):
        try:
            row = df_to_process.iloc[i]
            instance_id = row["instance"]
            name = row["names"]
            if isinstance(name, list):
                name = name[0]
            country = row["country"]
            language = COUNTRY_DICT[country]['languages'][0]
            tokens = custom_tokenizer(name, language)
            if tokens is None:
                tokenized_name = name
            else:
                tokenized_name = (" ".join(custom_tokenizer(name, language)))
            df_saved.loc[len(df_saved)] = [instance_id, name, tokenized_name]
        except Exception as e:
            print(f"Error processing instance {instance_id}: {e}")
            df_saved.loc[len(df_saved)] = [instance_id, name, name]

        # Save every 100 iterations
        if (i + 1) % 100 == 0 or (i + 1) == len(df_to_process):
            df_saved.to_csv(save_path, index=False)

    print("Tokenization complete!")
    return df_saved


############################################################# DECOMPOSER ######################################################
DB_PATH = "./data/wikdict_dbs/"


def rebuild_wikdict_databases():
    input_path = "./data/wikdict_dbs/input/"
    output_path = "./data/wikdict_dbs/output"
    languages = set()

    # Load country dictionary and extract valid languages
    valid_languages = {lang for data in COUNTRY_DICT.values() for lang in data.get("languages", [])}
    detected_languages = {entry.split('.')[0] for entry in os.listdir(input_path)}

    for lang in detected_languages & valid_languages:
        print(lang)
        make_db(lang, input_path, output_path)


def decompose_names(df, save_path="./results/decomposed_names.csv"):
    if os.path.exists(save_path):
        df_saved = pd.read_csv(save_path)
        processed_instances = set(df_saved["instance"])
    else:
        df_saved = pd.DataFrame(columns=["instance", "names", "decomposed"])
        processed_instances = set()

    df_to_process = df[~df["instance"].isin(processed_instances)]
    for i in tqdm(range(len(df_to_process)), desc="Decomposing Names"):
        try:
            row = df_to_process.iloc[i]
            instance_id = row["instance"]
            name = row["names"]
            if isinstance(name, list):
                name = name[0]  # Handle lists by taking the first name

            country = row["country"]
            language = COUNTRY_DICT.get(country, {}).get('languages', [None])[0]

            if language:
                decomposition = []
                for word in name.split(' '):
                    solution = split_compound(db_path='data/wikdict_dbs/output', lang=language, compound=word)
                    if solution:
                        decomposition.append(" ".join([part.written_rep for part in solution.parts]))
                    else:
                        decomposition.append(word)
                decomposed_name = " ".join(decomposition)
            else:
                decomposed_name = name  # Use original name if no language detected

            df_saved.loc[len(df_saved)] = [instance_id, name, decomposed_name]
        except Exception as e:
            print(f"Error processing instance {instance_id}: {e}")
            df_saved.loc[len(df_saved)] = [instance_id, name, name]  # Store original in case of error

        # Save every 100 iterations
        if (i + 1) % 100 == 0 or (i + 1) == len(df_to_process):
            df_saved.to_csv(save_path, index=False)

    print("Decomposition complete!")
    return df_saved



########################################################### RULE ALGORITHMS ######################################################
def word_counter_algorithm(names, labels, n=10):
    if sum(labels) == 0:
        #print('NO DATA')
        return []
    keyword_counts = Counter()

    # Update keyword counts based on labels
    for name, label in zip(names, labels):
        tokens = re.findall(r'\w+', str(name).lower())  # Lowercase for case insensitive matching
        # Counts
        if label:
            keyword_counts.update(tokens)
        else:
            keyword_counts.subtract(tokens)
    top_keywords = [kw for kw, _ in keyword_counts.most_common(n)]
    return top_keywords



def select_k_best_words(names, stpwrds, labels, n = 10):
    if sum(labels) == 0:
        #print('NO DATA')
        return []
    try:
        #Step 0: Initialize Vectorizer with no stopwords to preprocess simmilarly my stopword list
        vectorizer = TfidfVectorizer(stop_words=None)  # No stopwords yet
        normalized_stopwords = normalize_stopwords(stpwrds, vectorizer)

        # Step 1: Convert text to TF-IDF features = TfidfVectorizer(stop_words=normalized_stopwords, max_features=5000)
        vector_names = vectorizer.fit_transform(names)
        vector_names_list = vectorizer.get_feature_names_out()

        # Step 2: Apply SelectKBest with Chi-Square test
        selector = SelectKBest(score_func=chi2, k=n)
        selector.fit_transform(vector_names, labels)

        top_features = [vector_names_list[i] for i in selector.get_support(indices=True)]
        return top_features

    except Exception as e:
        print(f"Error: {e}")
        return []


def country_word_generator(names, countries, labels, method, ntokens=5):
    all_keywrds = {}

    df = pd.concat([names, countries, labels], axis=1)
    df.columns = ['names', 'country', 'labels']

    for country in COUNTRY_DICT.keys():
        df_country = df[df['country'] == country]
        country_names = df_country['names']
        country_labels = df_country['labels']
        country_keywrds = []
        if method == 'counter_algorithm':
            country_keywrds = word_counter_algorithm(country_names, country_labels, n=ntokens)
        elif method == 'idf_best':
            country_stpwrds = []
            languages = COUNTRY_DICT[country]['languages']
            for language in languages:
                if stopwords.has_lang(language):
                    country_stpwrds.extend(stopwords.stopwords(language))
            country_keywrds = select_k_best_words(country_names, country_stpwrds, country_labels, n=ntokens)
        all_keywrds[country] = country_keywrds
    return all_keywrds


############################################################# CLASSIFIER ######################################################
def rule_classify(names, keywords): # Receives a dict of country lists of regex patterns and matches names against it
    if keywords:
        patterns = r"(" + r"|".join(keywords) + r")"
        combined_pattern = re.compile(patterns, re.IGNORECASE)
        result = []
        for name in names:
            # Convert to string and handle NaN/None values
            if pd.isna(name):
                result.append(0)
            else:
                name_str = str(name)
                result.append(1 if combined_pattern.search(name_str) else 0)
        return result
    else:
        return [0] * len(names)
