# To detect the cells in a file and get the mask

Firstly activate the environment, keep the latest generated `.h5` file in the same folder, and rename it to `mask_rcnn_blood_final_model.h5`. 
Copy the `dataset` folder in the same folder, and also keep the test files in a same folder, say `data_test`, Note: The folder should only contain
**images** and no other files. Run the `python3 mask_rcnn4_generate_json.py data_test/` command and the `results.json` is generated in the `data_test` folder