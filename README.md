# Book Recommender

This project aims to build a simple text-based recommendation system for books using a dataset of 7k+ books with metadata. It leverages libraries such as **pandas** for data manipulation, **LangChain** and **Chroma** for similarity search, and **matplotlib** / **seaborn** for data exploration.

## Overview

1. **Data Source**  
   - The project uses dataset containing approximately 7,000 books’ metadata (e.g., ISBN, title, authors, categories, etc.).  

2. **Data Exploration**  
   - Reading and exploring the dataset with `pandas`:
     - Displaying the first few rows (`data.head()`).
     - Generating basic descriptive statistics (`data.describe()`).
   - Plotting distributions and inspecting features using `matplotlib` and `seaborn`.

3. **Embedding & Vector Store**  
   - **LangChain** and **Chroma** are used to create and store text embeddings:
     1. Each book’s description is combined with its ISBN and metadata into a single text field.
     2. **HuggingFaceEmbeddings** generates embeddings for each book’s text.
     3. **Chroma** stores the embeddings for fast similarity search.

4. **Recommendation Flow**  
   1. **Preprocessing**:  
      - Data is loaded, cleaned (if necessary), and relevant fields (description, title, ISBN, etc.) are concatenated into a single string.
      - These strings are turned into `Document` objects (LangChain format).
   2. **Embeddings**:  
      - `HuggingFaceEmbeddings` transforms each `Document` into a vector.
      - The vectors are indexed by **Chroma**, forming a searchable database.
   3. **Query Function**:  
      - A function `retrieve_recommendation(query, num_of_rec)` is defined to:
        1. Perform a similarity search over the Chroma database.
        2. Retrieve the top matching documents/ISBNs.
        3. Look up the matches in the original `pandas` DataFrame.
        4. Return a DataFrame of recommended books.

## Requirements

- **Python 3.7+**
- **pandas**, **seaborn**, **matplotlib**
- **langchain**, **chromadb**
- **kagglehub** (for dataset download)
- **HuggingFaceEmbeddings** (from LangChain’s huggingface integration)

## How To Run

1- Clone or Download this repository/notebook to your local machine.

2- Install the requirments
```bash
pip install -r requirements.txt
```
3- Then run each cell of this notebook
```bash
jupyter notebook recommender.ipynb
```
4- Execute cells in order to prepare data, build the vector store, and test with example queries
```bash
# For kids' book recommendations
retrieve_recommendation("recommend me childrens book?", 3)

# For nature-themed children's books
retrieve_recommendation("a book to teach children about nature?", 5)
```

