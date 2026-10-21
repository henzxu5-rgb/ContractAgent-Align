# Third-party data and model notices

## ContractNLI

Yuta Koreeda and Christopher D. Manning (2021), *ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts*, Findings of EMNLP 2021, pp. 1907–1919.

- Project and original dataset: https://stanfordnlp.github.io/contract-nli/
- Paper: https://aclanthology.org/2021.findings-emnlp.164/
- License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Original dataset SHA256: `e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b`.

This project separates contract text from annotations, preserves source span offsets, adapts the task into a tool environment, and creates trajectories, training examples, preference labels, and evaluation records. Copies and adaptations retaining ContractNLI content must preserve attribution and the applicable CC BY 4.0 notice. The MIT license for project source code does not replace the data license. The original authors do not endorse this project.

The source annotations were produced by researchers and crowd workers; paralegals participated in hypothesis review. No additional legal-expert or human preference annotation is claimed.

## Models and software

Base model: `Qwen/Qwen3-4B-Instruct-2507`, pinned revision `cdbee75f17c01a7cc42f958dc650907174af0554`; https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 . Base weights are not redistributed here. Users downloading models must consult their original model cards and license terms.

Teacher and Judge examples were generated through the DeepSeek API; service-returned model names are retained in the records. Model-generated preferences are AI feedback, not newly collected human feedback. Generated explanations have not been validated as professional legal advice.

Transformers, TRL, PEFT, PyTorch and other dependencies remain under their respective licenses. Development and experiment execution used AI coding assistance.
