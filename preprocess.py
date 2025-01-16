# standard
import numpy as np
import pandas as pd

# visualize
from tqdm.notebook import tqdm
from itertools import product

import multiprocessing

# system
import pickle     ## saving library
import os         ## file manager
import sys
from multiprocessing import Pool

import re         ## preprocessing text library
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')     # download toolkit for textblob.TextBlob.words

from textblob import TextBlob
from nltk.stem import PorterStemmer     # tranform expanding words of words like attacker, attacked, attacking -> attack
st = PorterStemmer()
from sklearn.model_selection import train_test_split

stop_words = stopwords.words('english')
stop = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "isnt", "it", "its", "itself", "keep", "keeps", "kept", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "names", "named", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "ok", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "puts", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "sees", "serious", "several", "she", "should", "show", "shows", "showed", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]




class PreprocessToolSimplified:
    def __init__(self, stop_words=None):
        self.stop_words = stop_words if stop_words else set()  # Initialize stop words

    def get_preprocessed_data(self, dataframe, 
                              preprocess_columns=['title', 'abstract', 'keywords'],
                              preprocessing_type=['Title', 'Abstract', 'Keywords'],
                              keep_columns=['itr'], n_jobs=4):
        import pandas as pd
        from multiprocessing import Pool
        import re
        
        # Ensure necessary columns are in the dataframe
        required_cols = set(preprocess_columns).union(keep_columns)
        missing_cols = required_cols - set(dataframe.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        # Define the output DataFrame structure
        output_data = pd.DataFrame(columns=preprocessing_type + keep_columns)
        output_data[preprocessing_type + keep_columns] = dataframe[preprocess_columns + keep_columns].copy()
        
        # Parallel processing
        with Pool(n_jobs) as pool:
            if 'Title' in preprocessing_type:
                output_data['Title'] = pool.map(self.preprocess_text, output_data['Title'])
            if 'Abstract' in preprocessing_type:
                output_data['Abstract'] = pool.map(self.preprocess_text, output_data['Abstract'])
            if 'Keywords' in preprocessing_type:
                output_data['Keywords'] = pool.map(self.preprocess_text, output_data['Keywords'])
            if 'Aims' in preprocessing_type and 'Aims' in dataframe.columns:
                output_data['Aims'] = pool.map(self.preprocess_text, output_data['Aims'])
        
        return output_data
    
    def preprocess_text(self, text):
        import re
        
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r"\d+", "", text)
            text = re.sub(r"[^a-z\s]", " ", text)
            filtered_sentence = [w for w in text.split() if w not in self.stop_words]
            text = " ".join(filtered_sentence).strip()
            text = re.sub(r"\s{2,}", " ", text)
        return text

def labelling_data(series, category):
    '''
    - Parameters:
        series (pandas Series): Conference distribution of data.
        category (Int64Index or list-like): Category (do not reset index of aims_content before using this function)
    - Returns:
        np.array: Label series for data
    '''
    label = np.zeros(len(series), dtype=int)
    for i, value in enumerate(category):
        label[series == value] = i
    return label


if __name__ == "__main__":
    tool_preprocess = PreprocessToolSimplified(stop_words) 
    # Sample usage
    # Working dir
    work_path = "./data"
    checkpoint_path = work_path + "checkpoint/"

    # create checkpoint path if not exists
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # Assuming work_path and checkpoint_path are defined

    # Function to split data into train, validate, and test sets
    def split_data(file_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        data = pd.read_csv(file_path, encoding="ISO-8859-1")
        train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio))
        val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)))
        return train_data, val_data, test_data

    # Walk through all files in the ./data folder
    data_folder = "./data"
    domain_lst = ['chemistry', 'biology', 'physics']
    for domain in domain_lst:
        path = os.path.join(data_folder, domain)
        for root, dirs, files in os.walk(path):
            print(f"Processing folder: {root}")
            print(f"Files: {files}")
            paper_file_name = ''
            aims_file_name = ''
            for file in files:
                if 'papers' in file:
                    paper_file_name = file
                if 'journal' in file:
                    aims_file_name = file
            file_path = os.path.join(root, paper_file_name)
            print(f"Processing file: {file_path}")
            
            # Split the data
            train_data, val_data, test_data = split_data(file_path)
            
            # Preprocess the data
            train_data = tool_preprocess.get_preprocessed_data(
                train_data,
                preprocess_columns=["title", "abstract", "keywords"],
                preprocessing_type=["Title", "Abstract", "Keywords"],
                keep_columns=["itr"],
                n_jobs=multiprocessing.cpu_count()
            )
            val_data = tool_preprocess.get_preprocessed_data(
                val_data,
                preprocess_columns=["title", "abstract", "keywords"],
                preprocessing_type=["Title", "Abstract", "Keywords"],
                keep_columns=["itr"],
                n_jobs=multiprocessing.cpu_count()
            )
            test_data = tool_preprocess.get_preprocessed_data(
                test_data,
                preprocess_columns=["title", "abstract", "keywords"],
                preprocessing_type=["Title", "Abstract", "Keywords"],
                keep_columns=["itr"],
                n_jobs=multiprocessing.cpu_count()
            )
           
            file_path_aims = os.path.join(root, aims_file_name)
            data_aims = pd.read_csv(file_path_aims, encoding = "ISO-8859-1")
            data_aims = tool_preprocess.get_preprocessed_data(
                data_aims,
                preprocess_columns = ["aims"],
                preprocessing_type = ["Aims"],
                keep_columns = ["itr"],
                n_jobs=multiprocessing.cpu_count()
                )

            # Save the preprocessed data
            preprocessed_data_path = checkpoint_path + 'preprocessed_data/'
            if not os.path.exists(preprocessed_data_path):
                os.makedirs(preprocessed_data_path)
                
            train_data['Label'] = labelling_data(train_data["itr"], data_aims["itr"])
            val_data['Label'] = labelling_data(val_data["itr"], data_aims["itr"])
            test_data['Label'] = labelling_data(test_data["itr"], data_aims["itr"])
            
            train_data.to_csv(preprocessed_data_path + f"{domain}_train.csv", index=False)
            val_data.to_csv(preprocessed_data_path + f"{domain}_validate.csv", index=False)
            test_data.to_csv(preprocessed_data_path + f"{domain}_test.csv", index=False)  
            data_aims.to_csv(preprocessed_data_path + f"{domain}_aims.csv", index=False)
            
    # print("Preprocessing train ....")
    # data_train = pd.read_csv(work_path + "/raw_data/data_splited_train.csv", encoding = "ISO-8859-1")
    # data_train = tool_preprocess.get_preprocessed_data(
    #     data_train,
    #     preprocess_columns = ["title", "abstract", "keywords"],
    #     preprocessing_type = ["Title", "Abstract", "Keywords"],
    #     keep_columns = ["itr"],
    #     n_jobs=multiprocessing.cpu_count()
    #     )

    # print("Preprocessing validate ....")
    # data_validate = pd.read_csv(work_path + "/raw_data/data_splited_validate.csv", encoding = "ISO-8859-1")
    # data_validate = tool_preprocess.get_preprocessed_data(
    #     data_validate,
    #     preprocess_columns = ["title", "abstract", "keywords"],
    #     preprocessing_type = ["Title", "Abstract", "Keywords"],
    #     keep_columns = ["itr"],
    #     n_jobs=multiprocessing.cpu_count()
    #     )

    # print("Preprocessing test ....")
    # data_test = pd.read_csv(work_path + "/raw_data/data_origin_test.csv", encoding = "ISO-8859-1")
    # data_test = tool_preprocess.get_preprocessed_data(
    #     data_test,
    #     preprocess_columns = ["title", "abstract", "keywords"],
    #     preprocessing_type = ["Title", "Abstract", "Keywords"],
    #     keep_columns = ["itr"],
    #     n_jobs=multiprocessing.cpu_count()
    #     )

    # print("Preprocessing aims ....")
    # data_aims = pd.read_csv(work_path + "/raw_data/aims_scopes.csv", encoding = "ISO-8859-1")
    # data_aims = tool_preprocess.get_preprocessed_data(
    #     data_aims,
    #     preprocess_columns = ["aims"],
    #     preprocessing_type = ["Aims"],
    #     keep_columns = ["itr"],
    #     n_jobs=multiprocessing.cpu_count()
    #     )

    # data_train['Label'] = labelling_data(data_train["itr"], data_aims["itr"])
    # data_validate['Label'] = labelling_data(data_validate["itr"], data_aims["itr"])
    # data_test['Label'] = labelling_data(data_test["itr"], data_aims["itr"])


    # preprocessed_data_path = checkpoint_path + 'preprocessed_data/'
    # if not os.path.exists(preprocessed_data_path):
    #     os.makedirs(preprocessed_data_path)

    # data_train.to_csv(preprocessed_data_path + "01_train.csv", index=False)
    # data_validate.to_csv(preprocessed_data_path + "01_validate.csv", index=False)
    # data_test.to_csv(preprocessed_data_path + "01_test.csv", index=False)
    # data_aims.to_csv(preprocessed_data_path + "01_aims.csv", index=False)
    
    
    # tool_preprocess = PreprocessToolSimplified(stop_words) 
    # # Sample usage
    # # Working dir
    # work_path = "./"
    # checkpoint_path = work_path + "checkpoint/"

    # # create checkpoint path if not exists
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    # # Assuming work_path and checkpoint_path are defined

    # def process_and_save_data(file_path, preprocess_columns, preprocessing_type, keep_columns, chunk_size=5000):
    #     chunk_number = 0
    #     for chunk in pd.read_csv(file_path, encoding="ISO-8859-1", chunksize=chunk_size):
    #         chunk = tool_preprocess.get_preprocessed_data(
    #             chunk,
    #             preprocess_columns=preprocess_columns,
    #             preprocessing_type=preprocessing_type,
    #             keep_columns=keep_columns,
    #             n_jobs=multiprocessing.cpu_count()
    #         )
    #         # chunk.to_csv(f"{file_path}_processed_chunk_{chunk_number}.csv", index=False)
    #         chunk_number += 1

    # print("Preprocessing train ....")
    # process_and_save_data(work_path + "/raw_data/data_splited_train.csv", ["title", "abstract", "keywords"], ["Title", "Abstract", "Keywords"], ["itr"])

    # print("Preprocessing validate ....")
    # process_and_save_data(work_path + "/raw_data/data_splited_validate.csv", ["title", "abstract", "keywords"], ["Title", "Abstract", "Keywords"], ["itr"])

    # print("Preprocessing test ....")
    # process_and_save_data(work_path + "/raw_data/data_origin_test.csv", ["title", "abstract", "keywords"], ["Title", "Abstract", "Keywords"], ["itr"])

    # print("Preprocessing aims ....")
    # process_and_save_data(work_path + "/raw_data/data_origin_test.csv", ["aims"], ["Aims"], ["itr"])
