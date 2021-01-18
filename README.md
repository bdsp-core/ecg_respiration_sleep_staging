# Sleep staging using ECG and/or respiratory effort

Available inputs:
* Electrocardiogram (ECG): `python test_SHHS.py ECG`
* Abdominal respiratory effort (ABD): `python test_SHHS.py ABD`
* Chest respiratory effort (CHEST): `python test_SHHS.py CHEST`
* ECG + ABD: `python test_SHHS.py ECG+ABD`
* ECG + CHEST: `python test_SHHS.py ECG+CHEST`

The example shows how to run these models on the [SHHS dataset](https://sleepdata.org/datasets/shhs).

Notes:
* due to github constraint, models exceeding 100MB cannot be pushed. Therefore they are split using [filesplit](https://github.com/ram-jayapalan/filesplit).
* to get models/CNN\_ECG+ABD\_fold1.pth, run `cd models`, and then `python merge_file.py CNN_ECG+ABD_fold1_to_be_merged`
* to get models/CNN\_ECG+CHEST\_fold1.pth, run `cd models`, and then `python merge_file.py CNN_ECG+CHEST_fold1_to_be_merged`
