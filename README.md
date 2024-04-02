# BiGen: Bigram Name Generator
A bigram language model trained to predict person names based on the given dataset. It generates names per character basis using the Feed forward neural network. The model is trained on datasets containing male and female names in the Gujarati language. However, it can be adapted to train on datasets in any language.

# Requirements
- Python 3.10
- PyTorch 2.1.2

# How To Run
- Clone repo: 
  git clone https://github.com/cancer32/BiGen.git
- Change directory:
  cd BiGen
- Install dependencies:
  pip install -r requirements.txt
- Generate an "output.txt" file containing the predictions:
  python generate.py --count 50 --weights ./weights/names_female.out

# Train The Model
- python train.py -i ./dataset/names_female.txt -o ./weights/names_female.out --seed 1 --learning_rate 50 --training_loop 1000
- python train.py -i ./dataset/names_male.txt -o ./weights/names_male.out --seed 1 --learning_rate 50 --training_loop 1000
