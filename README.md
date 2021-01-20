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

### If you use this in your work, please cite the following paper

[Sun, H., Ganglberger, W., Panneerselvam, E., Leone, M.J., Quadri, S.A., Goparaju, B., Tesh, R.A., Akeju, O., Thomas, R.J. and Westover, M.B., 2020. Sleep staging from electrocardiography and respiration with deep learning. Sleep, 43(7), p.zsz306.])https://doi.org/10.1093/sleep/zsz306)
