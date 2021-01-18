import os
import sys
from fsplit.filesplit import Filesplit

input_path = str(sys.argv[1]) # CNN_ECG+ABD_fold1.pth
output_dir = os.path.splitext(os.path.basename(input_path))[0]+'_to_be_merged'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
fs = Filesplit()
fs.split(file=input_path, split_size=1024*1024*50, output_dir=output_dir)
