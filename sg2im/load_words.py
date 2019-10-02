import numpy as np
import pdb


def main():
	glove_file = './sg2im/data/word_emb/glove.trimmed.300.npz'
	vocab_file ='./sg2im/data/word_emb/vocab.txt'
	embeddings = load_embs(glove_file)
	vocab_list = create_vocab_list(vocab_file)
    
    ## use cases
	test_words_list = ["water-other", "hair drier", "potted plant"]
	for i in range(len(test_words_list)):
		word = test_words_list[i]
		vocab_word, idx = map_to_vocab_list(word, vocab_list)
		word_emb = embeddings[idx]
		print(vocab_word, idx)
	

def get_vocab_word(word, vocab_list):
	vocab_word, idx = map_to_vocab_list(word, vocab_list)
	out = [vocab_word, idx]
	return tuple(out)
	#word_emb = embeddings[idx]
	#print(word, vocab_word, idx)

def get_embs(idx,  embeddings):
	return embeddings[idx, :]


def load_all_embs(glove_file):
	embeddingz = np.load(glove_file)
	embeddings = embeddingz['glove'] #embeddings will be a numpy array
	return embeddings

def create_vocab_list(vocab_file):
	vocab_list=[]	 
	with open (vocab_file) as f:
	    for line in f:
	        split_line = line.lstrip().rstrip().split(" ")
	        vocab_list.append(split_line[0])

	return vocab_list        


def check_exception_vocabs(word):
	# unchanged_word_list = ["potted plant", "stop sign", "solid-other", "water-drops"]
	# unchanged_word_list = ["solid-other", "water-drops"]
	# if word in unchanged_word_list:
	# 	vocab_word = word
	#el
	if word == "NONE":
		vocab_word = "background" #"none" #"background"
	elif word == "ski" or word == "skis":
		vocab_word = "ski"
	elif word == "playingfield":
		vocab_word = "field" 
	elif word == "wine glass":
		vocab_word = "wine"	
	elif word == "structural-other":
		vocab_word = "structure"
	elif word =="teddy bear":
		vocab_word = "teddy"
	elif word == "waterdrops":  
	 	vocab_word = "drops"
	# elif word == "potted plant":  
	# 	vocab_word = "plant"
	# elif word =="stop sign":  
	# 	vocab_word = "stop"
	elif word =="water-drops":  
		vocab_word = "drops"
	else:
		vocab_word = "NA"	
	return vocab_word			



def map_to_vocab_list(word, vocab_list):
	## check the exceptions first
	ret_word = check_exception_vocabs(word)


	if ret_word != "NA": 
		out = [ret_word, vocab_list.index(ret_word)]
		return tuple(out)


	## check two word categories and extract the second part for COCO objects
	if ' ' in word:
		ret_word = word.split(" ")[1]
		out = [ret_word, vocab_list.index(ret_word)]
		return tuple(out)
	
	## check hyphenated word and extract first part for COCO-stuff	
	elif '-' in word:	
		ret_word = word.split("-")[0]
		out = [ret_word, vocab_list.index(ret_word)]
		return tuple(out)
	

	## default : most COCO category names and VG names are unaltered in the dictionary
	ret_word = word

	if ret_word not in vocab_list:
		ret_word = "none"
		print(word, 'not found in embedding vocab_list')
	idx = vocab_list.index(word)
	out = [word, idx]
	return tuple(out)




if __name__ == '__main__':
  main()



	
