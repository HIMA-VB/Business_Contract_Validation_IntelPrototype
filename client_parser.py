import sys
from reader_parser import Reader  # Update import to use correct casing for class name

def main(file_path1, file_path2):
    reader_obj = Reader()
    documents1 = reader_obj.load_document(file_path1)
    documents2 = reader_obj.load_document(file_path2)

    print("Loaded Document 1: ")
    for document in documents1:
        print(document)

    print("Loaded Document 2: ")
    for document in documents2:
        print(document)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 client.py path_to_pdf_file1 path_to_pdf_file2")
    else:
        file_path1 = sys.argv[1]
        file_path2 = sys.argv[2]
        main(file_path1, file_path2)
