# EKG
This is the code for the paper EKG
# Preparation
To run the code, you first should download the embeddings of the words and the entities of the Wikipedia, where you can find them in https://nlp.stanford.edu/data/glove.6B.zip and http://openke.thunlp.org/. 

The code is implemented by Python 3.7, and the requirements of the enviroments are as follows:
<code>
  
beautifulsoup4==4.9.3
  
nltk==3.5
  
numpy==1.19.4
  
scikit-learn==0.23.2
  
scipy==1.5.4
  
spacy==3.1.0
  
tensorboard==1.11.0
  
tensorboard-plugin-wit==1.7.0
  
torch==1.1.0
  
torchsummary==1.5.1
  
torchtext==0.4.0
  
torchvision==0.3.0
  
tqdm==4.38.0
  
urllib3==1.26.2

  </code>

Then in the folder entity_model, run the command:

<code>
  python main.py --data_path /flickr_yelp/ --model model_with_entity --batch_size 8 --word_filter_threshold 0.8 --entity_filter_threshold 0.8 --n_epochs 50  --max_word_len_f 200 --max_word_len_t 200  --max_entity_length_f 200 --max_entity_length_t 200
  </code>

The meanings of the arguments are described in the main.py.   
