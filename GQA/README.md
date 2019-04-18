# GQA

## Data structure
```bash
├── Question Number
    ├── Annotations
    |   ├── answer
    |   ├── full Answer
    |   └── question
    │   
    ├── answer
    ├── entailed
    ├── equivalent
    ├── fullAnswer
    ├── groups
    ├── imageId
    ├── isBalanced
    ├── question
    ├── semantic
    ├── semanticStr
    └── types
        ├── detailed
        ├── semantic
        └── structural
```
- answer, groundtruth
- imageId
- question

## tqdm Module
```python
from tqdm import tqdm

for idx in tqdm(range(100), desc="test", mininterval=1):
    print(idx)
```
