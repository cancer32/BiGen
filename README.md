# make-gujarati-names
A bigram language model trained to pridict Gujarati names according to gender. It generates names per character basis using the probability of the next character

# Requirements
- Python 3.10
- PyTorch 2.1.2

# How to run
- Clone the repository.
- Change directory to make-gujarati-names
- Install dependencies: pip install -r requirements.txt
- python make_gujarati_names.py --count <number of names> --gender <male/female>
- This will output a text file "output.txt" containing the prediction