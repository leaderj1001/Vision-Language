# GQA: Visual Reasoning in the Real World
- [GQA dataset](https://cs.stanford.edu/people/dorarad/gqa/about.html)
- [GQA paper](https://arxiv.org/pdf/1902.09506.pdf)
- [GQA CVPR Workshop](https://visualqa.org/workshop.html)

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
- answer
- imageId
- question

## Network Architecture
![model](https://user-images.githubusercontent.com/22078438/57022053-eb349080-6c68-11e9-85a9-293ea120a1bd.PNG)

- Image Pretrained
  - [Tensornets github](https://github.com/taehoonlee/tensornets)
- Question Pretrained
  - [ELMo using tensorflow-hub](https://tfhub.dev/google/elmo/2)
- Attention model, We just use attention module
  - [Self-Attention Generative Adversarial Networks paper](https://arxiv.org/abs/1805.08318)
  - [Attention github](https://github.com/taki0112/Self-Attention-GAN-Tensorflow)

## Requirements
- tensorflow-gpu==1.13.1
- numpy==1.16.2
- tensorflow-hub==0.4.0
- python==3.7.3
- cv2==4.0.0
- tqdm==4.31.1
