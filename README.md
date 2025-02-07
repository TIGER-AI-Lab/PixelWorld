# PixelWorld
The official code of our paper [PixelWorld: Towards Perceiving Everything as Pixels](https://arxiv.org/abs/2501.19339).

Refactoring... There may be some problems with the reference relationship between codes.

## Installation
```bash
pip install -r requirements.txt
```

## Get Started
### From Local Files
```bash
python data.py --dataset WikiSS_QADataset --model GPT4o --mode text --prompt base
```
### From Huggingface Dataset
```bash
python data.py --dataset WikiSS_QADataset --model GPT4o --mode text --prompt base --from_hf
```

## Project Site
[PixelWorld](https://tiger-ai-lab.github.io/PixelWorld/)

## Citation
```
@article{lyu2024pixelworld,
    title={PixelWorld: Towards Perceiving Everything as Pixels},
    author={Lyu, Zhiheng and Ma, Xueguang and Chen, Wenhu},
    year={2025},
    eprint={2501.19339},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={http://arxiv.org/abs/2501.19339},
}
```
