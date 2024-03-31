# make-gujarati-names
A bigram language model trained to predict Gujarati names based on the trained dataset. It generates names per character basis using the Feed forward neural network

# Requirements
- Python 3.10
- PyTorch 2.1.2

# How To Run
- Clone repo: 
  git clone https://github.com/cancer32/make-gujarati-names.git
- Change directory:
  cd make-gujarati-names
- Install dependencies:
  pip install -r requirements.txt
- Generate an "output.txt" file containing the predictions
  python generate.py --count 50 --weights ./weights/names_female.out
  python generate.py --count 50 --weights ./weights/names_male.out

# Train The Model
- python train.py -i ./dataset/names_female.txt -o ./weights/names_female.out --seed 1 --learning_rate 50 --training_loop 1000
- python train.py -i ./dataset/names_male.txt -o ./weights/names_male.out --seed 1 --learning_rate 50 --training_loop 1000