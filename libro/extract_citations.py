import os
import re
import json

bib_file = '100_references.bib'
search_dir = '.'

# RegEx for Sphinx/MyST citations: {cite}`key1,key2` or {cite:p}`key1`
myst_cite_re = re.compile(r'\{cite[a-zA-Z:]*\}\`([^\`]+)\`')
# RegEx for pandoc/jupyter standard markdown: [@key1; @key2]
pandoc_cite_re = re.compile(r'\[(@[^\]]+)\]')

def get_bib_keys(path):
    keys = set()
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        # matches @article{key,
        matches = re.findall(r'@\w+\{([^,]+),', content)
        for m in matches:
            keys.add(m.strip())
    return keys

def extract_from_text(text):
    cited_keys = set()
    
    # MyST
    for match in myst_cite_re.finditer(text):
        keys_str = match.group(1)
        for k in keys_str.split(','):
            cited_keys.add(k.strip())
            
    # Pandoc
    for match in pandoc_cite_re.finditer(text):
        keys_str = match.group(1)
        # Split by ';' or ','
        parts = re.split(r'[;,]', keys_str)
        for p in parts:
            p = p.strip()
            if p.startswith('@'):
                cited_keys.add(p[1:])
                
    return cited_keys

def process_files():
    bib_keys = get_bib_keys(bib_file)
    all_cited = set()
    file_citations = {}

    for root, dirs, files in os.walk(search_dir):
        if '_build' in root or '.ipynb_checkpoints' in root:
            continue
        for f in files:
            if f.endswith('.md') or f.endswith('.ipynb'):
                if f == '100_references.bib': continue
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file_obj:
                        if f.endswith('.ipynb'):
                            data = json.load(file_obj)
                            text = ""
                            for cell in data.get('cells', []):
                                if cell.get('cell_type') == 'markdown':
                                    text += "".join(cell.get('source', [])) + "\n"
                        else:
                            text = file_obj.read()
                            
                    citations = extract_from_text(text)
                    if citations:
                        file_citations[path] = citations
                        all_cited.update(citations)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    
    print("=== SUMMARY ===")
    print(f"Total keys in bib: {len(bib_keys)}")
    print(f"Total unique cited keys: {len(all_cited)}")
    
    missing_in_bib = all_cited - bib_keys
    print(f"Missing in bib ({len(missing_in_bib)}): {sorted(list(missing_in_bib))}")
    
    unused_in_bib = bib_keys - all_cited
    print(f"In bib but not cited ({len(unused_in_bib)}): {sorted(list(unused_in_bib))}")

if __name__ == '__main__':
    process_files()
