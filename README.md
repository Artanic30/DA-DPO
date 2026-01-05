<div align="center">
  <h2>DA-DPO: Cost-efficient Difficulty-aware Preference Optimization for Reducing MLLM Hallucination</h2>

  <h2>TMLR 2025</h2>


</div>

<br></br>


This repository contains the reference code for the paper [NoisyGRPO: Incentivizing Multimodal CoT Reasoning via Noise Injection and Bayesian Estimatio](https://arxiv.org/abs/2510.21122).

[🎯 Project web page](https://artanic30.github.io/project_pages/DA-DPO/) |

[//]: # ([Paper]&#40;https://arxiv.org/pdf/2510.21122&#41; |)

[//]: # ([🤗 HuggingFace Model]&#40;https://huggingface.co/collections/Artanic30/noisygrpo&#41; |)


## Install Packages

```

conda create -n dadpo python=3.10 -y

conda activate dadpo

pip install -e .

```
## Training data
Download ShareGPT4V from [here](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)

Download COCO from [here](https://cocodataset.org/#home)

Download dataset annotation from [here](https://huggingface.co/datasets/renjiepi/BPO_Instruct)

Extract  data from ShareGPT4V and organize the images as follows:

```
Image_root
├── coco/
    ├──train2017/
├── llava/
    ├──llava_pretrain/
├── sam/
├── share_textvqa/
    ├──images/
├── web-celebrity/
    ├──images/
├── web-landmark/
    ├──images/
├── wikiart/
    ├──images/
```

## Train DADPO

```
bash scripts/da_dpo/7B_DADPO.sh
```


## Acknowledgement
The project is built on top of the amazing multimodal large language model [LLaVA](https://github.com/haotian-liu/LLaVA), RLHF package [trl](https://github.com/huggingface/trl), DPO for multimodal learning [Silkie](https://github.com/vlf-silkie/VLFeedback), visual contrastive decoding [VCD](https://github.com/DAMO-NLP-SG/VCD) and [BPO](https://github.com/pipilurj/bootstrapped-preference-optimization-BPO).
Thanks for these great work!


[//]: # (If you find our work useful for your research or applications, please cite using this BibTeX:)

[//]: # (```bibtex)

[//]: # (@misc{pi2024strengthening,)

[//]: # (      title={Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization},)

[//]: # (      author={Renjie Pi and Tianyang Han and Wei Xiong and Jipeng Zhang and Runtao Liu and Rui Pan and Tong Zhang},)

[//]: # (      year={2024},)

[//]: # (      eprint={2403.08730},)

[//]: # (      archivePrefix={arXiv},)

[//]: # (      primaryClass={cs.CL})

[//]: # (})

[//]: # (```)
