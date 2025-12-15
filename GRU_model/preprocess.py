import pandas as pd
import ast
import pickle
import os
from tqdm import tqdm

DATA_DIR = 'data'
OUTPUT_DIR = 'processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_products():
    print("Loading products...")
    # 1. Load New Site Products (The Target Vocabulary)
    new_products = pd.read_csv(os.path.join(DATA_DIR, 'new_site_products.csv'))
    new_products['id'] = new_products['id'].astype(int)
    
    # Base mapping: New Site Slug -> New Site ID
    slug_to_id = pd.Series(new_products.id.values, index=new_products.slug).to_dict()
    
    # 2. Load Old Site Mapping (Data Augmentation)
    # We want to map Old Site Slugs -> New Site IDs where a match exists
    try:
        mapping_df = pd.read_csv(os.path.join(DATA_DIR, 'old_site_new_site_products.csv'))
        old_products = pd.read_csv(os.path.join(DATA_DIR, 'old_site_products.csv'))
        
        # Map: Old ID -> New ID
        old_to_new_dict = pd.Series(mapping_df.new_site_id.values, index=mapping_df.old_site_id).to_dict()
        
        count_added = 0
        for _, row in old_products.iterrows():
            old_id = row['id']
            old_slug = row['slug']
            
            if old_id in old_to_new_dict:
                new_id = int(old_to_new_dict[old_id])
                
                # If this slug is not already in our map, add it
                if old_slug not in slug_to_id:
                    slug_to_id[old_slug] = new_id
                    count_added += 1
                    
        print(f"Augmented vocabulary with {count_added} old site slugs.")
        
    except FileNotFoundError:
        print("Warning: Old site mapping files not found. Skipping augmentation.")

    return slug_to_id, new_products['id'].unique()

def process_hits(file_name, slug_to_id):
    print(f"Processing {file_name}...")
    hits_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(hits_path):
        print(f"Warning: {hits_path} not found.")
        return {}
    
    chunksize = 100000
    watch_id_to_product = {}
    
    for chunk in tqdm(pd.read_csv(hits_path, chunksize=chunksize, usecols=['watch_id', 'page_type', 'slug'])):
        # Filter for PRODUCT pages
        product_hits = chunk[chunk['page_type'] == 'PRODUCT'].copy()
        
        # Map slug to product_id
        product_hits['product_id'] = product_hits['slug'].map(slug_to_id)
        
        # Drop hits with unknown products
        product_hits = product_hits.dropna(subset=['product_id'])
        
        # Store in dict
        for _, row in product_hits.iterrows():
            watch_id_to_product[row['watch_id']] = int(row['product_id'])
            
    return watch_id_to_product

def clean_sequence(sequence):
    """Removes consecutive duplicates (e.g., [A, A, B] -> [A, B])."""
    if not sequence:
        return []
    
    new_seq = [sequence[0]]
    for x in sequence[1:]:
        if x != new_seq[-1]:
            new_seq.append(x)
    return new_seq

def process_visits(file_name, watch_id_to_product, is_test=False):
    print(f"Processing {file_name}...")
    visits_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(visits_path):
        print(f"Warning: {visits_path} not found.")
        return []

    sessions = []
    
    # Columns to read
    usecols = ['visit_id', 'watch_ids']
    if not is_test:
        # We extract date_time for temporal sorting
        # We extract project_id just in case we want to filter later, 
        # but currently we accept both if they map to valid products.
        usecols.extend(['date_time', 'project_id'])
    
    chunksize = 50000
    
    for chunk in tqdm(pd.read_csv(visits_path, chunksize=chunksize, usecols=usecols)):
        for _, row in chunk.iterrows():
            visit_id = row['visit_id']
            
            try:
                watch_ids = ast.literal_eval(row['watch_ids'])
            except:
                watch_ids = []
                
            # Convert watch_ids to product sequence
            sequence = []
            for wid in watch_ids:
                try:
                    wid_int = int(wid)
                except:
                    continue
                    
                if wid_int in watch_id_to_product:
                    sequence.append(watch_id_to_product[wid_int])
            
            # Deduplicate consecutive items
            sequence = clean_sequence(sequence)

            # For test set, we must keep all sessions even if empty
            if len(sequence) > 0 or is_test:
                session_data = {
                    'visit_id': visit_id,
                    'sequence': sequence
                }
                if not is_test:
                    session_data['start_time'] = row['date_time']
                    # Optional: store project_id if we want to analyze distribution later
                    session_data['project_id'] = row['project_id']
                
                sessions.append(session_data)
                
    return sessions

def main():
    # 1. Load Product Mapping (New Site + Mapped Old Site)
    slug_to_id, all_product_ids = load_products()
    print(f"Total mapped products: {len(slug_to_id)}")
    
    # Save product IDs for vocabulary
    with open(os.path.join(OUTPUT_DIR, 'product_ids.pkl'), 'wb') as f:
        pickle.dump(all_product_ids, f)

    # 2. Process Training Data
    train_hits_map = process_hits('metrika_hits.csv', slug_to_id)
    train_sessions = process_visits('metrika_visits.csv', train_hits_map)
    
    print(f"Found {len(train_sessions)} training sessions (Old + New Site).")
    
    # Save training sessions
    with open(os.path.join(OUTPUT_DIR, 'train_sessions.pkl'), 'wb') as f:
        pickle.dump(train_sessions, f)
        
    # 3. Process Test Data
    test_hits_map = process_hits('metrika_hits_test.csv', slug_to_id)
    test_sessions = process_visits('metrika_visits_test.csv', test_hits_map, is_test=True)
    
    print(f"Found {len(test_sessions)} test sessions.")
    
    # Save test sessions
    with open(os.path.join(OUTPUT_DIR, 'test_sessions.pkl'), 'wb') as f:
        pickle.dump(test_sessions, f)

if __name__ == '__main__':
    main()
