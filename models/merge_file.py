import sys
from fsplit.filesplit import Filesplit

input_dir = str(sys.argv[1])# CNN_ECG+ABD_fold1_to_be_merged
    
fs = Filesplit()
fs.merge(input_dir=input_dir, output_file=input_dir.replace('_to_be_merged','.pth'))
