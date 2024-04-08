import csv
import pandas as pd
from operator import itemgetter

def process_csv(input_path, output_path):
    output = []
    with open(input_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header_skipped = False
                
        temp = []
        for row in csvreader:
            if not header_skipped:
                header_skipped = True
                continue  # Skip the first row (header) 
            current_str = row[2]    
            if any(char.isalpha() for char in current_str) or any(char.isdigit() for char in current_str):
                temp.append(row)
            else:
                if "." in current_str or "?" in current_str or "," in current_str:
                    temp.append(row)
                    output.append(temp)
                    temp = []
                else:
                    continue    

    with open(output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["start time", "end time", "utterance"]
        csvwriter.writerow(header)
        for row in output:
            sentence = ""
            for lis in row:
                sentence += lis[2]
            new_row = [row[0][0], row[-1][1], sentence]
            csvwriter.writerow(new_row)

def combine_lists(main_list, sub_list):
    # Create a new list to store the combined output
    combined_list = []

    # Iterate over each sublist in main_list along with its corresponding number from sub_list
    for sublist, number in zip(main_list, sub_list):
        # Append a new sublist with the number inserted as the third element (index 2)
        combined_sublist = sublist[:2] + [number] + sublist[2:]
        combined_list.append(combined_sublist)

    return combined_list

def combine_conversation(input_path1, input_path2, output_path):
    dtype_dict = {'start': int, 'end': int, 'text': str}
    df1 = pd.read_csv(input_path1, header=None, dtype=dtype_dict)
    # df1 = df1.apply(pd.to_numeric, errors='ignore') 
    df2 = pd.read_csv(input_path2, header=None, dtype=dtype_dict)
    # df2 = df2.apply(pd.to_numeric, errors='ignore')   
    df1 = df1.iloc[1:]
    df2 = df2.iloc[1:]
    conv1 = df1.values.tolist()
    conv2 = df2.values.tolist()
    conv1 = [["speaker 1"] + sent for sent in conv1] 
    conv2 = [["speaker 2"] + sent for sent in conv2] 

    print(type(conv1[0][1]))

    mid_points1 = [(sentence[1] + sentence[2])/2 for sentence in conv1]
    mid_points2 = [(sentence[1] + sentence[2])/2 for sentence in conv2]

    conv1 = combine_lists(conv1, mid_points1)
    conv2 = combine_lists(conv2, mid_points2)

    # 1 : start, 2 : mid, 3 : fin
    output = sorted(conv1+conv2, key=lambda x: x[1])
    # output = sorted(conv1+conv2, key=itemgetter(1))
    df = pd.DataFrame(output)
    df.columns = ["speaker", "start", "mid", "end", "text"]
    df.to_csv(output_path, index=False, header=True)
    return


input_path1 = "/Users/taichi/Documents/audio/speaker1.wav.csv"
output_path1 = "/Users/taichi/Documents/audio/sentences_speaker1.csv"
input_path2 = "/Users/taichi/Documents/audio/speaker2.wav.csv"
output_path2 = "/Users/taichi/Documents/audio/sentences_speaker2.csv"
# final_output = "/Users/taichi/Documents/audio/conversation_speaker1and2_by_mid_time.csv"

final_output = "/Users/taichi/Documents/audio/conversation_speaker1and2_by_start_time.csv"

# process_csv(input_path1, output_path1)
# process_csv(input_path2, output_path2)
# combine_conversation(output_path1, output_path2, final_output)

combine_conversation(input_path1, input_path2, final_output)

