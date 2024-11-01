# Coq Code Formatter Learning

This project implements a machine learning approach to learn and suggest formatting conventions for Coq code, based on the paper "Learning to Format Coq Code Using Language Models" by Nie et al.

## Overview

The project uses two approaches to learn Coq code formatting:
1. N-gram model: A simple statistical model based on token sequences
2. Neural model: A bidirectional LSTM network for more sophisticated formatting predictions

## Features

- Tokenization of Coq source code
- Space prediction between tokens
- Support for different coding styles (standard Coq and SSReflect/MathComp)
- Dataset statistics and analysis tools
- Model evaluation metrics

## Project Structure

```
coq-format-learning/
├── README.md
├── requirements.txt
├── data/
│   ├── sample_input/        # Sample Coq files
│   │   ├── simple.v         # Standard Coq style
│   │   └── mathcomp_style.v # MathComp style
│   └── processed/           # Processed token data
├── src/
│   ├── preprocessor.py      # Token extraction
│   ├── ngram_model.py      # N-gram model
│   ├── neural_model.py     # Neural model
│   └── utils.py            # Helper functions
└── train.py                # Training script
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Shift-Happens/coq-format-learning.git
cd coq-format-learning
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train models on sample data:
```bash
python train.py
```

2. Use specific files for training:
```bash
python train.py --input data/sample_input/simple.v
```

## Model Performance

Not assesed yet

## Contributing

Contributions are welcome! Here are some ways you can contribute:
- Add support for more Coq formatting features
- Improve the neural model architecture
- Add more sample Coq files
- Improve documentation
- Report issues

## License

MIT License

## Citation

If you use this code in your research, please cite:
```
@article{nie2020learning,
  title={Learning to Format Coq Code Using Language Models},
  author={Nie, Pengyu and Palmskog, Karl and Li, Junyi Jessy and Gligoric, Milos},
  journal={arXiv preprint arXiv:2006.16743},
  year={2020}
}
```

## Acknowledgments

This project is based on the research paper by Nie et al. from The University of Texas at Austin and KTH Royal Institute of Technology.