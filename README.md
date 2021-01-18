# Sleep staging using ECG and/or respiratory effort

Available inputs:
* Electrocardiogram (ECG)
* Abdominal respiratory effort (ABD)
* Chest respiratory effort (CHEST)
* ECG + ABD
* ECG + CHEST

The example shows how to run these models on the [SHHS dataset](https://sleepdata.org/datasets/shhs).

Notes:
* due to github constraint, models exceeding 100MB cannot be pushed. Therefore they are split using [filesplit](https://github.com/ram-jayapalan/filesplit).
* to get models/CNN\_ECG+ABD\_fold1.pth, run `cd models`, and then `python merge\_file.py CNN\_ECG+ABD\_fold1\_to\_be_merged`
* to get models/CNN\_ECG+CHEST\_fold1.pth, run `cd models`, and then `python merge\_file.py CNN\_ECG+CHEST\_fold1\_to\_be_merged`
