from src.pci_crackdown_functions import * 

proc_embedding(
    input_file = "data/input/sgns_renmin_Word+Character+Ngram/sgns_renmin_Word+Character+Ngram.txt",
    output_path = "data/output/embeddings"
)

proc_data(
    data_path = 'data/input/Tiananmen_sentences.pkl', 
    embedding_path = 'data/output/embeddings/embedding.pkl', 
    tokenizer_path = 'data/output/embeddings/tokenizer.pkl', 
    create_training_sample = 1,
    output_path = 'data/output/',
    filename = "tam.pkl"
)

proc_data(
    data_path = 'data/input/HK2014_sentences.pkl', 
    embedding_path = 'data/output/embeddings/embedding.pkl', 
    tokenizer_path = 'data/output/embeddings/tokenizer.pkl', 
    create_training_sample = 0,
    output_path = 'data/output/',
    filename = "prediction_data_HK2014.pkl"
)

proc_data(
    data_path = 'data/input/HK2019_sentences.pkl', 
    embedding_path = 'data/output/embeddings/embedding.pkl', 
    tokenizer_path = 'data/output/embeddings/tokenizer.pkl', 
    create_training_sample = 0,
    output_path = 'data/output/',
    filename = "prediction_data_HK2019.pkl"
)