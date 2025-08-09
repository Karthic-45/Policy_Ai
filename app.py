def load_documents(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyMuPDFLoader(file_path).load()
    elif ext in [".doc", ".docx", ".pptx", ".html", ".htm"]:
        return UnstructuredFileLoader(file_path).load()
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
        return [Document(page_content=df.to_string())]
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path).load()
    elif ext == ".eml":
        return UnstructuredEmailLoader(file_path).load()
    elif ext in [".png", ".jpg", ".jpeg"]:
        return UnstructuredImageLoader(file_path).load()
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return [Document(page_content=df.to_string())]
    elif ext == ".zip":
        docs = []
        with tempfile.TemporaryDirectory() as extract_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        try:
                            docs.extend(load_documents(full_path))
                        except Exception as e:
                            print(f"⚠ Skipped file in ZIP: {file} ({e})")
        return docs
    else:
        # Last resort: try unstructured for unknown binary formats
        try:
            return UnstructuredFileLoader(file_path).load()
        except Exception as e:
            raise ValueError(f"Unsupported or unreadable file extension: {ext} ({e})")


