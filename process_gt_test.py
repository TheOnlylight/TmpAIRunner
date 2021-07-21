import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import csv
import os


def hex2dec(hex):
    """Convert a hexadecimal string to a decimal number"""
    hex = "0x" + hex
    result_dec = int(hex, 0)
    return result_dec


def generate_pulse_gt(file_path):
    gt_time = []          # an empty list to store the first column
    gt_pulse = []         # an empty list to store the second column
    call_in = False
    with open(file_path, 'r') as rf:
        reader = csv.reader(rf, delimiter=',')
        for row in reader:
            if (int(row[0]) % 1000000 < 150):
                call_in = True
            
            if call_in:
                gt_time.append(int(row[0]) % 1000000 + 1000000) # for the timestamp, only last 6 numbers will change over time
            else:
                gt_time.append(int(row[0]) % 1000000)
                
                
            gt_pulse.append(hex2dec(row[1]))
    return gt_pulse, gt_time


def process_back_gt(CSV_FILE_PATH):
    print('Back Finger CSV_FILE_PATH: ', CSV_FILE_PATH)
    print("df")
    try:
        df = pd.read_csv(CSV_FILE_PATH, header=None)
    except Exception as e:
        print("Broken file!")
        with open(CSV_FILE_PATH) as csv_file:
          csv_reader = csv.reader(csv_file, delimiter=',')
          csvFile = open(CSV_FILE_PATH[:-4] + '_Fixed.csv', 'w', newline='', encoding='utf-8')
          csv_writer = csv.writer(csvFile)
          former_row = []
          num = 0
          for row in csv_reader:
              # print(row)
              # print(len(row))
      
              if (len(row) == 5 and not (row[0] == '' or row[1] == '' or row[2] == '' or row[3] == '' or row[4] == '' or len(row[0]) != 13)):
                  csv_writer.writerow([str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])])
                  former_row = row
              else:
                  csv_writer.writerow([str(former_row[0]), str(former_row[1]), str(former_row[2]), str(former_row[3]), str(former_row[4])])
      
          csvFile.close()
      
        os.rename(CSV_FILE_PATH, CSV_FILE_PATH[:-4]+'_Old.csv')
        os.rename(CSV_FILE_PATH[:-4] + '_Fixed.csv', CSV_FILE_PATH)
        df = pd.read_csv(CSV_FILE_PATH, header=None)
            
    print("df0", df.shape)
    df = df.to_numpy()
    
    ts = []
    call_in = False
    #print("df1", df.shape)
    for i in df:
        if (int(i[0]) % 1000000 < 150):
            call_in = True
        
        if call_in:
            ts.append(int(i[0]) % 1000000 + 1000000)
        else:
            ts.append(int(i[0]) % 1000000)
    
    ts=np.array(ts)        
    #ts = df[:, 0] % 1000000 # Timestamp
    df = df[:, -3] # Red channel
    #print("df2", df.shape)
    frame_num = df.shape[0] // 1296 # 1296 = 36 * 36
    df = df[:frame_num*1296]
    #print("df3", df.shape)
    df = np.reshape(df, [frame_num, 1296, 1])
    #print("df4", df.shape)
    df = np.average(df, axis=1)
    #print("df5", df.shape)
    ts = ts[:frame_num*1296]
    ts = np.reshape(ts, [frame_num, 1296, 1])
    ts = np.average(ts, axis=1)

    return df, ts
