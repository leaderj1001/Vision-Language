# Usage of Custom DataLoader

## Advantage of DataLoader(My opinion)
- Because the DataLoader has a number of paramters, it can be very handy if you only create Datasets.

## Usage
- Most existing data is data that requires preprocessing. Also, the data capacity is often larger than expected.
- So, we use the pickle module to preprocess and store the data.
- This example was done using a GQA dataset.
  - [Data URL](https://cs.stanford.edu/people/dorarad/gqa/download.html)

1. Make pkl data file using pickle module
```python
import glob
import json
import pickle

def get_file_names():
  file_names = glob.glob("D:/GQA/questions1.2/train/*.json")
  return file_names

def load_json(file_path):
  with open(file_path, "r") as f:
      data = json.load(f)
  return data

def write_pickle(data, filename):
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(data, f)
    
# make pickle data file
file_names = get_file_names()
file_names = [file_names[0]]

pickle_list = []

for file_name in file_names:
    print(file_name)
    data = load_json(file_name)

    for line in data:
        # print(line, data[line]["answer"], data[line]["imageId"], data[line]["question"])
        pickle_list.append([data[line]["answer"], data[line]["imageId"], data[line]["question"]])
write_pickle(pickle_list, "test")
```

2. Configure your own custom Dataset class.
  - [Example of DataLoader](https://github.com/leaderj1001/Vision-Language/blob/master/Pytorch-DataLoader-Example/main.py)
