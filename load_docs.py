from pypdf import PdfReader
import os

docs_folder = "docs"
all_text = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(docs_folder, filename)
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        all_text.append(text)
        print(f"Loaded {filename}: {len(reader.pages)} pages")

print(f"\nTotal documents loaded: {len(all_text)}")
print("\nSample text from first document:")
print(all_text[0][:500])