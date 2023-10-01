


ALL_FOLDERS = ["DA_UnAug_1"]


counter = 0

for folder_name in ALL_FOLDERS:
    if counter%1==0:    
        print(folder_name,"----")
    # read the contents of COMPLEXITY_DUMP.txt from each folder
    
    complexity_dump_path = folder_name + "/COMPLEXITY_DUMP.txt"
    
    with open(complexity_dump_path, "r") as f:
        # Iterate through each line in the file
        for line in f:
            # Split the line by ":"
            parts = line.strip().split(":")
            
            # Check if there are at least two parts
            if len(parts) >= 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                # Print the key and value except for the last line
                if key != "FLOPs (total float operations)":
                    if key == "Total Parameters" or key == "Trainable params" or key == "Non-trainable params":
                        # print("Key == ",key)
                        value = float(value)/1e6
                    if key != "Total Convolutional Layers" and key != "Total Linear (Dense) Layers":
                        value_param = "{:.4f}".format(float(value))
                    else:
                        value_param = int(value)
                    # use 4 decimal places on the value
                    print(value_param, end=" & ")
                    # print(value, end=" ")
    
    
    # Calculate the trianing time
    train_Epoch_time_dump_path = folder_name + "/TRAIN_EPOCH_TIME.txt"
    # Open the file for reading
    with open(train_Epoch_time_dump_path, "r") as file:
        # Read the lines and convert them to float values, skipping the first line
        lines = file.readlines()[1:]
        train_epoch_times = [float(line.strip()) for line in lines]

    # Calculate the mean and standard deviation
    mean = sum(train_epoch_times) / len(train_epoch_times)
    import math
    std_deviation = math.sqrt(sum((x - mean) ** 2 for x in train_epoch_times) / len(train_epoch_times))

    # Print the results with 4 decimal places
    print(f"{mean:.4f}","$\pm$",f"{std_deviation:.4f}",end=" & ")
    
    
    try:
        # Inference time
        inference_time_dump_path = folder_name + "/INFERENCE_TIME.txt"
        with open(inference_time_dump_path, "r") as file:
            # Read the lines and convert them to float values
            inference_times = [float(line.strip()) for line in file]

        # Calculate the mean and standard deviation
        mean = sum(inference_times) / len(inference_times)
        import math
        std_deviation = math.sqrt(sum((x - mean) ** 2 for x in inference_times) / len(inference_times))

        # Print the results with 4 decimal places
        print(f"{mean:.4f}","$\pm$",f"{std_deviation:.4f}")
        
    except:
        print("folder name", folder_name)
        pass
    
    print()
    
    
    
    counter += 1
    