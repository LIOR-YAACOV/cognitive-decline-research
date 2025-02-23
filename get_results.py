import os
import numpy as np

if __name__ == "__main__":
    root_dir = "new_res_files" # these files are result files from the server
    res_files = os.listdir("new_res_files")
    for file in res_files:
        train_acc = []
        val_acc = []
        with open(os.path.join("new_res_files",file), encoding="utf8") as f:
            for line in f.readlines():
                if "train loss:" in line:
                    train_acc.append(float(line.split(" ")[-1].split("\\n")[0]))
                if "validation loss:" in line:
                    val_acc.append(float(line.split(" ")[-1].split("\\n")[0]))

            train_acc = np.array(train_acc)
            val_acc = np.array(val_acc)

            min_index = np.argmin(val_acc)
            model_name = "_".join(file.split(".")[0].split("_")[3:])
            print(f"model {model_name} has min val_acc of {val_acc[min_index]:.2f}")
            print(f"model {model_name} has min train_acc of {train_acc[min_index]:.2f}\n")