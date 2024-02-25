# make-gujarati-names
A bigram language model trained to predict Gujarati names according to gender. It generates names per character basis using the probability of the next character

# Requirements
- Python 3.10
- PyTorch 2.1.2

# How to run
- Clone repo: 
  git clone https://github.com/cancer32/make-gujarati-names.git
- Change directory:
  cd make-gujarati-names
- Install dependencies:
  pip install -r requirements.txt
- python make_gujarati_names.py --count 50 --gender female
- This will output a text file "output.txt" containing the prediction