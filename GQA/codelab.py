# import json
# import codecs
#
# with codecs.open('./test.json', "r", encoding="utf-8", errors="ignore") as f:
#     d = json.load(f)
#     print(d)
#     for t in d:
#         print(d[t]["lon"])

# from tqdm import tqdm
#
# for idx in tqdm(range(100), desc="test", mininterval=1):
#     print(idx)

sentence = "Hello, world. Hi, I'm Myeongjun."
import re
re_sc = re.compile('[\!@#$%\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')

s = re_sc.sub(' ', sentence).strip().split()
print(s)
words = [w.strip() for w in s]
print(words)
words = [w for w in words if len(w) >= 1 and len(w) < 16]

test = []
if not words:
    test = [None] * 2

import mmh3
from collections import Counter
hash_func = lambda x: mmh3.hash(x, seed=17)
x = [hash_func(w) % 100001 for w in words]
print(x)
xv = Counter(x).most_common(16)
print(xv)
import numpy as np
max_len = 16
x = np.zeros(max_len, dtype=np.float32)

for i in range(len(xv)):
    x[i] = xv[i][0]
print(x)

print(test)
from collections import Counter

x = [4, 3, 3, 2, 1]
print(Counter(x))
print(Counter(x).most_common(3))
