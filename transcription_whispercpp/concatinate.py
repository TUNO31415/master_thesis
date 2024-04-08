import csv

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
        for row in output:
            sentence = ""
            for lis in row:
                sentence += lis[2]
            new_row = [row[0][0], row[-1][1], sentence]
            csvwriter.writerow(new_row)

input_path = "/Users/taichi/Documents/audio/speaker1.wav.csv"
output_path = "/Users/taichi/Documents/audio/sentences_speaker1.csv"

process_csv(input_path, output_path)