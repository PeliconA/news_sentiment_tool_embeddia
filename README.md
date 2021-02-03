# News Sentiment Analysis Tool

The tool classifies articles into negative, neutral or positive. It was developed using methods described in the following article.

@article{pelicon2020zero,
  title={Zero-shot learning for cross-lingual news sentiment classification},
  author={Pelicon, Andra{\v{z}} and Pranji{\'c}, Marko and Miljkovi{\'c}, Dragana and {\v{S}}krlj, Bla{\v{z}} and Pollak, Senja},
  journal={Applied Sciences},
  volume={10},
  number={17},
  pages={5993},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}

The tool is free to use and the code is open source. If you use the tool, please cite the article above.

## Installation

To install the tool, create a virtual environment using your preferred virtual environment tool for python and install the following dependencies:

- torch
- pandas
- numpy
- keras
- tqdm
- transformers
- sklearn

For further instructions of how to use the tool, chech the example script 'example.py'

Outputs:

The tool outputs a sentiment class for each of your input articles where '0' means the article is classified as negative, '1' means the article is classified as neutral and '2' means the article is classified as positive.
