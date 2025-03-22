import json

with open("data/country_dictionary.json", "r") as f:
    COUNTRY_DICT = json.load(f)

with open("data/entity_dictionary.json", "r") as f:
    ENTITY_DICT = json.load(f)



SPACY_MODELS = {
    'es': 'es_core_news_md',  # Spanish
    'ca': 'ca_core_news_md',  # Catalan
    'pt': 'pt_core_news_md',  # Portuguese
    'fr': 'fr_core_news_md',  # French
    'el': 'el_core_news_md',  # Greek
    'it': 'it_core_news_md',  # Italian
    'de': 'de_core_news_md',  # German
    'nl': 'nl_core_news_md',  # Dutch
    'sv': 'sv_core_news_md',  # Swedish
    'fi': 'fi_core_news_md',  # Finnish
    'lt': 'lt_core_news_md',  # Lithuanian
    'pl': 'pl_core_news_md',  # Polish
    'ro': 'ro_core_news_md',  # Romanian
    'sl': 'sl_core_news_md',  # Slovenian
    'da': 'da_core_news_sm',  # Danish
    'hr': 'hr_core_news_md'   # Croatian
}



DOMAIN_CLASSES_CORR = {
        'medical': ['hospital', 'university_hospital'],
        'administrative': ['local_government'],
        'education': ['primary_school', 'secondary_school']
    }

STRUCTURE_MAPPING = {
    'medical': ['3-class', 'nested-class'],
    'administrative': ['2-class'],
    'education': ['3-multiclass']
}

NLI_MODELS = {
        "roberta-large": "joeddav/xlm-roberta-large-xnli",
        "bge-m3": "MoritzLaurer/bge-m3-zeroshot-v2.0",
        "mDeBerta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "MiniLM": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    }

EMB_MODELS = {
        'multilingual-e5': {'model_name': 'intfloat/multilingual-e5-large-instruct', 'max_length': 512, 'batch_size': 512},
        'qwen': {'model_name': 'Alibaba-NLP/gte-Qwen2-7B-instruct', 'max_length': 8192, 'batch_size': 32},
        'mistral': {'model_name': 'intfloat/e5-mistral-7b-instruct', 'max_length': 4096, 'batch_size': 64},
        'e5-small': {'model_name': 'intfloat/e5-small-v2', 'max_length': 512, 'batch_size': 1024},
    }