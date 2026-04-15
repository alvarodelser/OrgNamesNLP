# %% [markdown]
# # Eurostat vs Wikidata Coverage Comparison
# 
# This notebook estimates the real-world number of:
# - Primary school students (proxy for primary schools)
# - Secondary school students (proxy for secondary schools)
# - Hospital beds (proxy for hospitals)
# - LAU units (proxy for local governments)
# 
# Then compares with the Wikidata entity counts extracted for this project.

# %%
import requests
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Project country mapping: Wikidata Q-code -> ISO flag code (Eurostat uses 2-letter ISO)
with open('./data/country_dictionary.json', 'r') as f:
    country_dict = json.load(f)

# Build mapping: ISO 2-letter code -> Country name, and Q-code -> ISO
iso_to_country = {}
qcode_to_iso = {}
iso_to_qcode = {}
for qcode, info in country_dict.items():
    iso = info['flag']
    iso_to_country[iso] = info['country']
    qcode_to_iso[qcode] = iso
    iso_to_qcode[iso] = qcode

# The EU countries we track in our project
TARGET_ISOS = sorted(iso_to_country.keys())
print(f"Target countries ({len(TARGET_ISOS)}): {TARGET_ISOS}")

# %% [markdown]
# ## 1. Eurostat API Helper

# %%
def fetch_eurostat_json(dataset_code, params):
    """
    Fetch data from the Eurostat Statistics API (JSON-stat format).
    Returns the raw JSON response.
    """
    base_url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset_code}"
    params['format'] = 'JSON'
    params['lang'] = 'EN'
    
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()


def eurostat_json_to_df(data):
    """
    Parse Eurostat JSON-stat 2.0 response into a DataFrame.
    The values are indexed by a flat position, which maps to the
    cartesian product of all dimensions (ordered by 'id' list, 'size' list).
    """
    dimensions = data['id']  # ordered dimension names
    sizes = data['size']     # sizes per dimension
    values = data.get('value', {})
    statuses = data.get('status', {})
    
    # Build category labels for each dimension
    dim_labels = {}
    for dim in dimensions:
        cat = data['dimension'][dim]['category']
        # index maps code -> position
        index = cat['index']
        label = cat.get('label', {})
        # Build list ordered by position
        ordered = sorted(index.items(), key=lambda x: x[1])
        dim_labels[dim] = [code for code, pos in ordered]
    
    # Generate all index combinations
    import itertools
    all_combos = list(itertools.product(*[dim_labels[d] for d in dimensions]))
    
    rows = []
    for flat_idx, combo in enumerate(all_combos):
        val = values.get(str(flat_idx))
        status = statuses.get(str(flat_idx))
        row = {dim: combo[i] for i, dim in enumerate(dimensions)}
        row['value'] = val
        row['status'] = status
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

# %% [markdown]
# ## 2. Fetch Secondary School Students (Upper + Lower)
# 
# - `educ_uoe_enrs04`: Upper secondary (ISCED 3 = ED3)
# - `educ_uoe_enrs01`: Lower secondary (ISCED 2 = ED2)
# 
# We want: total students, all sectors, both sexes, full headcount, for each country, year 2022 (latest complete).

# %%
# Build geo filter string with all target countries
geo_filter = TARGET_ISOS

def get_secondary_students(year='2022'):
    """Fetch upper + lower secondary students for all target countries."""
    
    results = {}
    
    # Upper secondary (ISCED 3)
    print("Fetching upper secondary (ISCED 3)...")
    data_upper = fetch_eurostat_json('educ_uoe_enrs04', {
        'isced11': 'ED3',
        'sex': 'T',
        'worktime': 'TOTAL',
        'sector': 'TOT_SEC',
        'geo': geo_filter,
        'time': year
    })
    df_upper = eurostat_json_to_df(data_upper)
    df_upper = df_upper[df_upper['value'].notna()]
    
    for _, row in df_upper.iterrows():
        geo = row['geo']
        if geo in TARGET_ISOS:
            results[geo] = results.get(geo, 0) + int(row['value'])
    
    # Lower secondary (ISCED 2)
    print("Fetching lower secondary (ISCED 2)...")
    data_lower = fetch_eurostat_json('educ_uoe_enrs01', {
        'isced11': 'ED2',
        'sex': 'T',
        'worktime': 'TOTAL',
        'sector': 'TOT_SEC',
        'geo': geo_filter,
        'time': year
    })
    df_lower = eurostat_json_to_df(data_lower)
    df_lower = df_lower[df_lower['value'].notna()]
    
    for _, row in df_lower.iterrows():
        geo = row['geo']
        if geo in TARGET_ISOS:
            results[geo] = results.get(geo, 0) + int(row['value'])
    
    print(f"  Got secondary student data for {len(results)} countries")
    return results

secondary_students = get_secondary_students('2022')
print("\nSecondary students per country:")
for iso in sorted(secondary_students.keys()):
    print(f"  {iso} ({iso_to_country.get(iso, iso)}): {secondary_students[iso]:,}")

# %% [markdown]
# ## 3. Fetch Primary School Students

# %%
def get_primary_students(year='2022'):
    """Fetch primary school students (ISCED 1) for all target countries."""
    
    print("Fetching primary school students (ISCED 1)...")
    data = fetch_eurostat_json('educ_uoe_enrp04', {
        'isced11': 'ED1',
        'sex': 'T',
        'worktime': 'TOTAL',
        'sector': 'TOT_SEC',
        'geo': geo_filter,
        'time': year
    })
    df = eurostat_json_to_df(data)
    df = df[df['value'].notna()]
    
    results = {}
    for _, row in df.iterrows():
        geo = row['geo']
        if geo in TARGET_ISOS:
            results[geo] = int(row['value'])
    
    print(f"  Got primary student data for {len(results)} countries")
    return results

primary_students = get_primary_students('2022')
print("\nPrimary students per country:")
for iso in sorted(primary_students.keys()):
    print(f"  {iso} ({iso_to_country.get(iso, iso)}): {primary_students[iso]:,}")

# %% [markdown]
# ## 4. Fetch Hospital Beds

# %%
def get_hospital_beds(year='2021'):
    """
    Fetch total hospital beds per country from hlth_rs_bds1.
    Facility = 'HBEDT' (Total hospital beds), unit = 'NR' (Number).
    Year 2021 typically has the most complete data.
    """
    
    print(f"Fetching hospital beds (year={year})...")
    data = fetch_eurostat_json('hlth_rs_bds1', {
        'unit': 'NR',
        'facility': 'HBEDT',
        'geo': geo_filter,
        'time': year
    })
    df = eurostat_json_to_df(data)
    df = df[df['value'].notna()]
    
    results = {}
    for _, row in df.iterrows():
        geo = row['geo']
        if geo in TARGET_ISOS:
            results[geo] = int(row['value'])
    
    print(f"  Got hospital bed data for {len(results)} countries")
    return results

hospital_beds = get_hospital_beds('2021')
print("\nHospital beds per country:")
for iso in sorted(hospital_beds.keys()):
    print(f"  {iso} ({iso_to_country.get(iso, iso)}): {hospital_beds[iso]:,}")

# %% [markdown]
# ## 5. Count LAU (Local Administrative Units) per Country
# 
# Download the official EU LAU Excel from Eurostat and count entries per country.

# %%
def get_lau_counts():
    """
    Download the LAU Excel file from Eurostat and count LAU per country.
    The file has a sheet with LAU data containing a CNTR_CODE column.
    """
    
    lau_url = "https://ec.europa.eu/eurostat/documents/345175/501971/EU-27-LAU-2025-NUTS-2024.xlsx/574c9e4a-2dae-99fe-5510-3fd18d8e90c2?t=1771601900130"
    
    print("Downloading LAU Excel file...")
    lau_path = './data/lau_2025.xlsx'
    
    if not os.path.exists(lau_path):
        response = requests.get(lau_url)
        response.raise_for_status()
        with open(lau_path, 'wb') as f:
            f.write(response.content)
        print(f"  Downloaded to {lau_path}")
    else:
        print(f"  Using cached {lau_path}")
    
    # Read the Excel file - the data is typically in the first or second sheet
    # Try to find the right sheet with LAU codes
    xls = pd.ExcelFile(lau_path)
    print(f"  Sheets: {xls.sheet_names}")
    
    # Usually the data sheet is named something like 'LAU' or 'EU-27'
    # Try reading the first sheet that has data
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(lau_path, sheet_name=sheet_name, header=0)
        # Look for column that contains country codes
        cols_lower = [str(c).lower() for c in df.columns]
        
        cntr_col = None
        for i, c in enumerate(cols_lower):
            if 'cntr' in c or 'country' in c:
                cntr_col = df.columns[i]
                break
        
        if cntr_col is not None:
            print(f"  Found country column '{cntr_col}' in sheet '{sheet_name}'")
            counts = df[cntr_col].value_counts()
            results = {}
            for iso in TARGET_ISOS:
                if iso in counts.index:
                    results[iso] = int(counts[iso])
                # Handle special cases like EL for Greece
                elif iso == 'GR' and 'EL' in counts.index:
                    results[iso] = int(counts['EL'])
            
            print(f"  Got LAU counts for {len(results)} countries")
            return results
    
    print("  WARNING: Could not find country column in LAU file")
    return {}

lau_counts = get_lau_counts()
print("\nLAU units per country:")
for iso in sorted(lau_counts.keys()):
    print(f"  {iso} ({iso_to_country.get(iso, iso)}): {lau_counts[iso]:,}")

# %% [markdown]
# ## 6. Load Wikidata Entity Counts

# %%
def get_wikidata_counts():
    """Load Wikidata hierarchy CSVs and count unique entities per country."""
    
    with open('./data/entity_dictionary.json', 'r') as f:
        entity_dict = json.load(f)
    
    categories = {
        'wd_hospitals': entity_dict['hospital'],      # Q16917
        'wd_local_gov': entity_dict['local_government'],  # Q6501447
        'wd_primary_schools': entity_dict['primary_school'],  # Q9842
        'wd_secondary_schools': entity_dict['secondary_school']  # Q159334
    }
    
    results = {}
    for label, qid in categories.items():
        filepath = f'./data/wikidata_{qid}_hierarchy.csv'
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Count unique instances per country
            counts = df.groupby('country')['instance'].nunique()
            for qcode, iso in qcode_to_iso.items():
                if qcode in counts.index:
                    if iso not in results:
                        results[iso] = {}
                    results[iso][label] = int(counts[qcode])
        else:
            print(f"  Warning: {filepath} not found")
    
    return results

wikidata_counts = get_wikidata_counts()
print("Wikidata entity counts loaded for", len(wikidata_counts), "countries")

# %% [markdown]
# ## 7. Combine Everything into a Summary Table

# %%
# Build the combined DataFrame
rows = []
for iso in TARGET_ISOS:
    country_name = iso_to_country.get(iso, iso)
    wd = wikidata_counts.get(iso, {})
    
    row = {
        'Country': country_name,
        'ISO': iso,
        # Eurostat data
        'Eurostat Primary Students': primary_students.get(iso),
        'Eurostat Secondary Students': secondary_students.get(iso),
        'Eurostat Hospital Beds': hospital_beds.get(iso),
        'Eurostat LAU Count': lau_counts.get(iso),
        # Wikidata data
        'Wikidata Primary Schools': wd.get('wd_primary_schools', 0),
        'Wikidata Secondary Schools': wd.get('wd_secondary_schools', 0),
        'Wikidata Hospitals': wd.get('wd_hospitals', 0),
        'Wikidata Local Governments': wd.get('wd_local_gov', 0),
    }
    rows.append(row)

summary_df = pd.DataFrame(rows)
summary_df = summary_df.set_index('Country')

# Sort by Eurostat Hospital Beds descending
summary_df = summary_df.sort_values('Eurostat Hospital Beds', ascending=False, na_position='last')

print("\n" + "="*120)
print("COMBINED EUROSTAT vs WIKIDATA COMPARISON")
print("="*120)
display(summary_df)

# %% [markdown]
# ## 8. Totals

# %%
# Show totals
totals = summary_df.drop(columns=['ISO']).sum()
print("\nTotal across all countries:")
print(totals.to_string())

# %%
# Save to CSV
output_path = './data/eurostat_wikidata_comparison.csv'
summary_df.to_csv(output_path)
print(f"\nSaved comparison table to {output_path}")
