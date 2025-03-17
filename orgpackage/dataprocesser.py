from SPARQLWrapper import SPARQLWrapper, JSON
import time
import os
import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
import ast

TIMEOUT = 15


sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)
sparql.addParameter("User-Agent", "European Public Authorities Classification (alvaro.fontecha@upm.es)")

def is_valid_json(raw_data):
    try:
        json.loads(raw_data)
        return True
    except json.JSONDecodeError:
        return False


def clean_errors(raw_data): #Aux function to clean wikidata return when error interrupts query
    first_item_index = raw_data.find('"item"')
    last_item_index = raw_data.rfind('"item"')
    # print(raw_data[last_item_index-10:last_item_index+6])
    if last_item_index != first_item_index:
        if is_valid_json(raw_data):
            return raw_data
        else:
            cropped_data = raw_data[:last_item_index - 10]  # Remove up to the comma before last valid item
            clean_data = cropped_data + "]}}"
            return clean_data
    else:
        return None

def raw_wikidata_instance_query(country_id, class_id, offset=0): #Queries wikidata for all instances of subclasses of organizations per country and stores clean raw result in jsons
    with open('./data/country_dictionary.json', 'r', encoding='utf-8') as f:
        COUNTRIES_DICT = json.load(f)
    print("Querying data from " + COUNTRIES_DICT[country_id]["country"])
    instance_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT * WHERE {{
          ?item wdt:P31/wdt:P279* wd:{class_id};
                wdt:P17 wd:{country_id};
        }}
        LIMIT 100000
        OFFSET 
        """
    sparql.setQuery(instance_query + str(offset))
    try:
        query_result = sparql.query()
        time.sleep(TIMEOUT)
    except Exception as e:
        print(f"Error: {str(e)[:100]}. Retrying after {str(TIMEOUT)} seconds...")
        time.sleep(TIMEOUT)
        try:
            query_result = sparql.query()
        except Exception as e2:
            print(f"Error: {str(e2)[:100]}. Omitting...")
            return
    raw_result = query_result.response.read().decode("utf-8", errors="replace")
    clean_result = clean_errors(raw_result)
    if clean_result is not None:
        write_path = "./data/raw/"+class_id+"/instances_" + country_id + ".json"
        if os.path.exists(write_path):
            with open(write_path, "r", encoding="utf-8") as file:  # Adding to the existing dictionary.
                previous_results = file.read()
                file.close()
            with open(write_path, "w", encoding="utf-8") as file:  # Crop results to append inside the bindings list
                first_item_index = clean_result.find("item")
                second_item_index = clean_result.find("item", first_item_index + 1)
                appendable_result = clean_result[second_item_index - 1:]
                complete_results = previous_results[:-3] + ", {" + appendable_result
                file.write(complete_results)
        else:
            with open(write_path, "w", encoding="utf-8") as file:
                file.write(clean_result)

def extract_wikidata_instances(class_id = 'Q43229'): # Queries wikidata based on lines already downloaded. Iterative process.
    with open('./data/country_dictionary.json', 'r', encoding='utf-8') as f:
        COUNTRIES_DICT = json.load(f)
    for country_id in COUNTRIES_DICT.keys():
        if not os.path.exists("./data/raw/"+class_id+"/instances_" + country_id + ".json"):
            os.makedirs("./data/raw/"+class_id+"/", exist_ok=True)
            raw_wikidata_instance_query(country_id, class_id)
        else:
            # print("Procesing: " + COUNTRIES_DICT[country_id]["country"])
            # instances = read_raw_instance_results(class_id=class_id)
            # offset = len(instances.loc[instances["country"] == country_id]) # Get the number of extracted lines
            # if offset > 1000000:
                print("Threshold reached, skipping...")
            # else:
            #     print("Lines before: " + str(offset))
            #     raw_wikidata_instance_query(country_id, class_id, offset=offset)
            #     instances = read_raw_instance_results(class_id=class_id)
            #     lines = len(instances.loc[instances["country"] == country_id])
            #     print("Lines after: " + str(lines))

def read_raw_instance_results(class_id = 'Q43229'): #Reads raw files into a df
    instance_query_results = []
    folder_path = f"./data/raw/{class_id}/"
    for filename in os.listdir(folder_path):
        if filename.startswith("instances_") and filename.endswith(".json"):
            country_id = filename.split("_")[1].split(".")[0]  # Extract country ID from filename
            read_path = os.path.join(folder_path, filename)
            with open(read_path, "r", encoding="utf-8", errors="replace") as file:
                clean_data = file.read()
            parsed_data = json.loads(clean_data)["results"]["bindings"]
            for result in parsed_data:
                instance = {
                    "instance": result["item"]["value"],
                    "country": country_id
                }
                instance_query_results.append(instance)
    instance_query_df = pd.DataFrame(instance_query_results).drop_duplicates(subset=['instance'], keep='first')
    return instance_query_df


def plot_data_volume(dfs, labels=None):
    # Convert single DataFrame to a list
    if not isinstance(dfs, list):
        dfs = [dfs]
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(dfs))]

    # Compute the counts for each DataFrame
    country_counts_list = [df['country'].value_counts() for df in dfs]

    # Get the unique set of countries across all datasets
    all_countries = sorted(set().union(*[set(counts.index) for counts in country_counts_list]), key=lambda country: -sum(counts.get(country, 0) for counts in country_counts_list))


    # Map country codes to names and flags
    with open('./data/country_dictionary.json', 'r', encoding='utf-8') as f:
        COUNTRIES_DICT = json.load(f)
    country_flags = {key: value["flag"].lower() for key, value in COUNTRIES_DICT.items()}
    country_names = {key: value["country"] for key, value in COUNTRIES_DICT.items()}

    fig, ax = plt.subplots(figsize=(10, 10))

    # Bar width settings
    num_datasets = len(dfs)
    bar_width = 0.8 / max(num_datasets, 1)  # Avoid division by zero

    # Color cycle
    colors = plt.cm.tab10.colors

    # Define the cap height
    max_visible_height = 20000

    # Plot bars for each dataset
    for i, (country_counts, label) in enumerate(zip(country_counts_list, labels)):
        bar_positions = np.arange(len(all_countries)) - (0.4 - i * bar_width * num_datasets) / max(num_datasets, 1)
        actual_heights = [country_counts.get(country, 0) for country in all_countries]
        heights = [min(h, max_visible_height) for h in actual_heights]  # Cap the bar height at 20,000

        bars = ax.bar(bar_positions, heights, width=bar_width, color=colors[i % len(colors)], label=label)

        # Add text labels inside or above bars
        for bar, height, actual_height in zip(bars, heights, actual_heights):
            if actual_height > max_visible_height or actual_height < 100:
                text_y = bar.get_y() + height + 100  # Position text just above the bar
                ax.text(bar.get_x() + bar.get_width() / 2, text_y, f"{int(actual_height)}",
                        ha="center", va="bottom", fontsize=5, color="black", rotation=90)

    ax.set_ylim(-1000, max_visible_height)
    ax.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax.axhline(0, color='black', linewidth=1)
    # Add flag images (centered correctly)
    for idx, country_id in enumerate(all_countries):
        flag_code = country_flags.get(country_id)
        if flag_code:
            png_image_path = os.path.join("../country-flags/flags/png", flag_code + ".png")
            try:
                flag_img = Image.open(png_image_path).resize((25, 15))
                flag_img = np.array(flag_img)

                # Compute flag position
                x_pos = idx if num_datasets > 1 else np.arange(len(all_countries))[idx] - 0.5
                y_pos = -800

                imagebox = OffsetImage(flag_img, zoom=0.5)
                ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False, box_alignment=(0.5, 0))
                ax.add_artist(ab)

            except FileNotFoundError:
                print(f"Flag image for {country_names.get(country_id, country_id)} not found!")

    # Set x-axis labels
    ax.set_xticks(range(len(all_countries)))
    ax.set_xticklabels([country_names.get(country, country) for country in all_countries], rotation=60, ha="right")

    ax.set_xlabel('Country')
    ax.set_ylabel('Number of Rows')
    ax.legend()

    plt.show()


def plot_data_classes(final_df):
    plots = {
        "Medical": ['hospital', 'university_hospital'],
        "Administrative": ['local_government'],
        "Education": ['primary_school', 'secondary_school']
    }
    for key in plots:
        df_list = [final_df] + [final_df[final_df[label] == 1] for label in plots[key]]
        labels = ['Total'] + plots[key]
        plot_data_volume(df_list, labels)

def most_common_classes(df):
    all_classes = []
    for classes_list in tqdm(df['classes'], desc="Flattening lists", unit="row"):
        all_classes.extend(classes_list)  # Extend the main list with the elements of each list
    class_counts = Counter(all_classes)
    sorted_class_counts = class_counts.most_common()
    return sorted_class_counts

def sample_instances(instance_query_df, sample_size = 25000): #Samples df per country
    countries_dfs = []
    with open('./data/country_dictionary.json', 'r', encoding='utf-8') as f:
        COUNTRIES_DICT = json.load(f)
    for country in COUNTRIES_DICT.keys():
        country_df = instance_query_df[instance_query_df["country"] == country]
        if len(country_df) > sample_size:
            country_df = country_df.sample(n=sample_size, random_state=42)
        countries_dfs.append(country_df)
    sampled_df = pd.concat(countries_dfs)
    sampled_df.to_csv("./data/wikidata_organization_instances_sample"+str(sample_size)+".csv", index=False)

def generate_clean_class_df(sample_path, class_path): #CAUTION. Will overwrite results.
    instance_df = pd.read_csv(sample_path)
    instance_df["names"] = "Unknown"
    instance_df["class_ids"] = "Unknown"
    instance_df["classes"] = "Unknown"
    instance_df.to_csv(class_path, index=False)

def extract_wikidata_classes(class_file, english_label = False): #In batches, queries wikidata obtaining label and class info per instance in class_df
    class_df = pd.read_csv(class_file)
    original_rows = len(class_df)
    unprocessed_df = class_df[class_df["classes"] == "Unknown"]

    names_dict = {}  # Dictionary to store names
    classes_dict = {}
    class_ids_dict = {}  # Dictionary to store class IDs
    batch_size = 200  # Define batch size

    with open('./data/country_dictionary.json', 'r', encoding='utf-8') as f:
        COUNTRIES_DICT = json.load(f)
    for country in COUNTRIES_DICT.keys():
        country_df = unprocessed_df[unprocessed_df["country"] == country]
        country_instances = ["wd:" + row["instance"].split("/")[-1] for _, row in country_df.iterrows()]
        print(len(country_instances))
        languages = COUNTRIES_DICT[country]["languages"]
        languages_str = ', '.join([f'"{lang}"' for lang in languages])  # Format languages properly

        # Process instances in batches
        for i in tqdm(range(0, len(country_instances), batch_size),
                      desc=f"Processing {COUNTRIES_DICT[country]['country']}"):
            instances_str = " ".join(country_instances[i:i + batch_size])
            if english_label:
                class_query = f"""
                                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                    PREFIX wd: <http://www.wikidata.org/entity/>
                                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                                    SELECT ?item (GROUP_CONCAT(DISTINCT ?name; SEPARATOR=", ") AS ?names) 
                                                 (GROUP_CONCAT(DISTINCT ?class; SEPARATOR=", ") AS ?class_ids) 
                                                 (GROUP_CONCAT(DISTINCT ?classLabel; SEPARATOR=", ") AS ?classes) 
                                    WHERE {{
                                      VALUES ?item {{ {instances_str} }}
                                      ?item wdt:P31 ?class;
                                            rdfs:label ?name.
                                      FILTER(LANG(?name) = "en")
                                      ?class rdfs:label ?classLabel.
                                      FILTER(LANG(?classLabel) = "en")
                                    }}
                                    GROUP BY ?item
                                """
            else:
                class_query = f"""
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
                    SELECT ?item (GROUP_CONCAT(DISTINCT ?name; SEPARATOR=", ") AS ?names) 
                                 (GROUP_CONCAT(DISTINCT ?class; SEPARATOR=", ") AS ?class_ids) 
                                 (GROUP_CONCAT(DISTINCT ?classLabel; SEPARATOR=", ") AS ?classes) 
                    WHERE {{
                      VALUES ?item {{ {instances_str} }}
                      ?item wdt:P31 ?class;
                            rdfs:label ?name.
                      FILTER(LANG(?name) IN ({languages_str}))
                      ?class rdfs:label ?classLabel.
                      FILTER(LANG(?classLabel) = "en")
                    }}
                    GROUP BY ?item
                """
            # print(class_query)
            try:
                sparql.setQuery(class_query)
                results = sparql.query().convert()

                for res in results["results"]["bindings"]:
                    item = res["item"]["value"]
                    names_dict[item] = res.get("names", {}).get("value", "")
                    class_ids_dict[item] = res.get("class_ids", {}).get("value", "")
                    classes_dict[item] = res.get("classes", {}).get("value", "")

                # Update the DataFrame progressively
                class_df["class_ids"] = class_df["instance"].map(class_ids_dict).fillna(class_df["class_ids"])
                class_df["names"] = class_df["instance"].map(names_dict).fillna(class_df["names"])
                class_df["classes"] = class_df["instance"].map(classes_dict).fillna(class_df["classes"])


                assert len(class_df) == original_rows, "Error: Rows have been lost!"
                class_df.to_csv(class_file, index=False)
            except Exception as e:
                print(f"Error: {str(e)[:50]}. Continuing after {str(TIMEOUT)} seconds...")
                time.sleep(TIMEOUT)


def obtain_subhierarchy(class_id):
    extract_wikidata_instances(class_id=class_id)
    read_raw_instance_results(class_id=class_id).to_csv("./data/wikidata_"+class_id+"_hierarchy.csv")
    generate_clean_class_df("./data/wikidata_"+class_id+"_hierarchy.csv", "./data/wikidata_"+class_id+"_hierarchy.csv")
    extract_wikidata_classes("./data/wikidata_"+class_id+"_hierarchy.csv", english_label=False)
    extract_wikidata_classes("./data/wikidata_"+class_id+"_hierarchy.csv", english_label=True)

def consolidate_hierarchy(df):
    class_counter = defaultdict(int)
    class_instances = defaultdict(set)  # Using set to avoid duplicate instances
    for _, row in df.iterrows():
        instance_names = row["names"]
        for class_label in row["classes"]:  # Each row has a list of classes
            class_counter[class_label] += 1
            class_instances[class_label].add(instance_names[0])  # Store instances for each class

    prominent_classes_df = pd.DataFrame(
        {
            "class": list(class_counter.keys()),
            "count": list(class_counter.values()),
            "instances": [list(name) for name in class_instances.values()],  # Convert sets to lists
        }
    )
    prominent_classes_df = prominent_classes_df.sort_values(by="count", ascending=False).reset_index(drop=True)
    print(prominent_classes_df)

def load_dataset(filename):
    df = pd.read_csv(filename)
    df['class_ids'] = df['class_ids'].apply(ast.literal_eval)
    df['classes'] = df['classes'].apply(ast.literal_eval)
    return df

def enricher():
    # Step 1: Read the main dataset
    org_df = load_dataset('data/wikidata_organizations_dataset.csv')
    class_labels = ['hospital', 'university_hospital', 'local_government', 'primary_school', 'secondary_school']

    # Step 1.5: Sample dynamically per country according to the number of positive instances of each domain, summed.
    with open('./data/country_dictionary.json', 'r', encoding='utf-8') as f:
        COUNTRIES_DICT = json.load(f)
    cutoffs = {country: 0 for country in COUNTRIES_DICT}
    for label in class_labels:
        if label != 'university_hospital': # All u_hospitals are hospitals so they are accounted for
            for country in COUNTRIES_DICT.keys():
                file_path = os.path.join('data', f'wikidata_{label}_dataset.csv')
                if os.path.exists(file_path):
                    aux_df = load_dataset(file_path)
                    aux_df = aux_df[aux_df['country'].isin(COUNTRIES_DICT.keys())]
                cutoffs[country] += min(aux_df[aux_df['country']==country].shape[0], 5000)
    cutoffs = {country: cutoff * 2 for country, cutoff in cutoffs.items()} # Twice the sample to ensure at least the same amount of negative instances.

    def get_top_rows(group, cutoff):
        return group.head(cutoff)
    org_df = org_df.groupby('country').apply(lambda x: get_top_rows(x, cutoffs[x.name])).reset_index(drop=True)

    for label in class_labels:
        # Step 2: Generate label columns for each class
        org_df[label] = org_df['classes'].apply(lambda x: 1 if label in x else 0)

        file_path = os.path.join('data', f'wikidata_{label}_dataset.csv')
        if os.path.exists(file_path):
            aux_df = load_dataset(file_path)
            aux_df = aux_df[aux_df['country'].isin(COUNTRIES_DICT.keys())]

            # Step 3: If auxiliary dataset for a country has more than 5000 samples, downsample to 5000
            def sample_per_country(group, threshold):
                if len(group) > threshold:
                    return group.sample(n=threshold, random_state=42)
                return group
            aux_df = aux_df.groupby('country').apply(lambda x: sample_per_country(x, 5000)).reset_index(drop=True)

            # Step 4: Merging Datasets
            # 4.1 - Ensure the auxiliary dataset has the same columns as the original dataset (add missing columns with 0)
            for col in org_df.columns:
                if col not in aux_df.columns:
                    aux_df[col] = 0  # Add missing columns with default value of 0
            print('aux ' + label)
            print(aux_df.shape[0])
            # 4.2 - For duplicates, update the column in org_df and continue
            for _, row in org_df.iterrows():
                instance = row['instance']
                if instance in aux_df['instance'].values:
                    org_df.loc[org_df['instance'] == instance, label] = 1
                    aux_df = aux_df[aux_df['instance'] != instance]
            # 4.3 - Set the label column to 1
            aux_df[label] = 1
            # 4.4 - Concatenate aux_df with org_df

            print('before '+label)
            print(org_df[org_df[label]==1].shape[0])
            org_df = pd.concat([org_df, aux_df], ignore_index=True)
            print('after ' + label)
            print(org_df[org_df[label]==1].shape[0])
            # Step 5: Remove duplicates based on 'instance', keeping the org_df rows. There should not be any.
            #org_df = org_df.sort_values(by=[label], ascending=True).drop_duplicates(subset='instance', keep='first')
    # Step 6 we will keep only the first name. Another way to augment data is to split names into several instances but we lose the unique identifier. This could yield benefits when dealing with several coofficial languages but could be detrimental in low data scenarios where names could be overrepresented
    org_df['names'] = org_df['names'].apply(ast.literal_eval)
    org_df["names"] = org_df["names"].apply(lambda x: x[0])
    return org_df
