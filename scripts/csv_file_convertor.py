############################################################################################################
#   To run the file, run the following first line below in terminal:                                       #
#                                                                                                          #
#   python main.py --csv_path <path_to_csv> --textfile_path <path_to_output_textfile>                      #
#                                                                                                          #
#   After pressing Enter, you will be prompted to select which mode you want, Write mode or append mode.   #
#   Either press W or A. Defaults to Write mode.                                                           #
#   Write mode will create a new text file. Append mode will allow you to add to already existing file.    #
############################################################################################################

from copy import copy
import pandas as pd
import argparse
import os
import glob

def read_df(path):
    return pd.read_csv(path) 

def get_df_with_required_cols(df, req_cols=None):
    if req_cols is None:
        req_cols = ['seeker_post', 'response_post', 'rationales', 'episode_done']
    new_df = None
    if isinstance(req_cols, list):
        try:
            new_df = df[req_cols]
        except KeyError:
            print(f"One of the keys {req_cols} not there in the input dataset")
    return new_df

def write_to_training_file(df, output_text_file_path='output.txt', mode='w'):
    with open(output_text_file_path, mode, newline='') as f:
        for index, row in df.iterrows():
            seeker_text = row['seeker_post']
            response_text = row['response_post']
            rationales_text = row['rationales']
            if isinstance(rationales_text, str):
                response_text = response_text + " " + rationales_text.replace("|", " ").strip()

            if episode_done := row['episode_done']:
                f.write(f"text:{seeker_text}\tlabels:{response_text}\tepisode_done:{episode_done}")
            else:
                f.write(f"text:{seeker_text}\tlabels:{response_text}")
            if index + 1 < len(df):
                f.write("\n")

def split_to_test_and_validation(input_file_path):
    test_file = 'test.txt'
    valid_file = 'valid.txt'
    text_list = []
    with open(input_file_path, 'r', newline='') as f:
        text_list =f.readlines()
    test_index = len(text_list) // 10
    with open(test_file, "w") as test_f:
        test_f.write("".join(text_list[:test_index]))
    with open(valid_file, "w") as valid_f:
        valid_f.write("".join(text_list[test_index:2*test_index]))

def convert_to_input_csv(input_file_path):
    """
    Returns a dataframe formatted with the required columns from the input csv.
    Decides which is the reponse and seeker_post based on the type and will consider the first row as a seeker_post
    """
    if os.path.isdir(input_file_path):
        csv_files = glob.glob(os.path.join(input_file_path, "*.csv"))
        df = pd.concat([pd.read_csv(f) for f in csv_files])
    else:
        df = pd.read_csv(input_file_path)
    persona = None
    conversation_id = str(df["ID"].iloc[0]).split("_")[0]
    text = ""
    output_list = []
    output = []
    row_iterator = df.iterrows()
    _, current_row = next(row_iterator)  # take first item from row_iterator
    for _, next_row in row_iterator:
        # This is for the first message
        conversation_id_prefix = next_row["ID"].split("_")[0] 
        if persona is None:
            text = current_row["Utterance"]
            persona = current_row["Type"]
            continue
        if persona != current_row["Type"]:
            # if the other person start talking add the previous person text to the output list
            output.append(text)
            if len(output) == 2:
                output.append("") # This is for the rationales which is empty
                episode_done = False
                if conversation_id != conversation_id_prefix:
                    episode_done = True
                    conversation_id = conversation_id_prefix
                output.append(episode_done)
                output_list.append(copy(output))
                output = []
            text = current_row["Utterance"]
        else:
            # If the next message is from the same person append it to the previous
            text = text + " " + current_row["Utterance"]
        persona = current_row["Type"]
        current_row = next_row

    return pd.DataFrame(output_list, columns=["seeker_post", "response_post", "rationales", "episode_done"])


if __name__ == '__main__':
    req_cols = ['seeker_post', 'response_post', 'rationales', 'episode_done'] #Required columns must be there. The column names of the csv file.
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to the input CSV file.') #csv path
    parser.add_argument('--textfile_path', type=str, help='Path to the output text file in ParlAI Dialogue format).') #output text file path
    args = parser.parse_args()
    
    mode = input('Please press A for append mode or W for Write mode. Defaults to WRITE mode.')
    mode = mode.lower() if mode.lower() in ['a', 'w'] else 'w'
    
    output_file_path = "train.txt" if args.textfile_path is None else args.textfile_path
    path = "train.csv" if args.csv_path is None else args.csv_path
    if not os.path.exists(path):
        print(f"dir or file with name {path} does not exist")
        os._exit(1)


    df = convert_to_input_csv(path)
    new_df = get_df_with_required_cols(df, req_cols)
    print(new_df)
    if new_df is not None:
        write_to_training_file(new_df, output_text_file_path=output_file_path, mode=mode.strip().lower())
    else:
        print("Some problem. Cannot create the dataset.")

    split_to_test_and_validation(output_file_path)
    
