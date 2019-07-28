import os
import re
import pandas as pd 
import numpy as np 
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from gensim.models import word2vec 
from sklearn import naive_bayes, svm, preprocessing 
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier as RFC 
from sklearn.model_selection import cross_val_score 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection.univariate_selection import chi2, SelectKBest 

# parameters

vector_type = "TFIDF"
model_name = "GoogleNews-vectors-negative300.bin"
model_type = "bin"

# parameters for word2vec
#num_featues need to be identical with the pre-trianed model
num_featues = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3 

# training model = {"RF", "NB", "SVM", "BT", "no"}
training_model = "NB"

# Feature scaling = {"Standard", "signed", "unsigned", "no"}
# scaling is needed for SVM
scaling = "no"


# dimension reduction = {"SVD", "chi2", "no"}
# for NB models, we cannot perform truncate SVD as it will make input negative
# chi is the feature selection based on chi2 independence test
dim_reduce = "chi2"
num_dim = 500




# functions 

def clean_text(raw_text, remove_stopwords = False, output_format = "string"):
	# remove HTML markup
	text = BeautifulSoup(raw_text, features = "html.parser")
	# keep only characters
	text = re.sub("[^a-zA-Z]", " ", text.get_text())

	# split words and store to list
	text = text.lower().split()

	if remove_stopwords:
		# use set of it has 0(1) lookup time
		stops = set(stopwords.words("english"))
		words = [w for w in text if w not in stops]

	else:
		words = text

	# Return a cleaned string or list
	if output_format == "string":
		return " ".join(words)

	elif output_format == "list":
		return words



def text_to_doublelist(text, tokenizer, remove_stopwords = False):
	raw_sentences = tokenizer.tokenize(text.strip())
	sentence_list = []

	for raw_sentence in raw_sentences:
		sentence_list.append(clean_text(raw_sentence, False, "list"))
	return sentence_list

def text_to_vec(words, model, num_features):
	feature_vec = np.zeros((num_features), dtype = "float32")
	word_count = 0
	index2word_set = set(model.index2word)

	for word in words:
		if word in index2word_set:
			word_count += 1
			feature_vec += model[word]

	feature_vec /= word_count 
	return feature_vec 


def gen_text_vecs(texts, model, num_features):
	cur_index = 0
	text_features_vecs = np.zeros((len(texts), num_features), dtype = "float32")
	for text in texts:
		if curr_index % 1000 == 0.:
			print("Vectorizing text %d of %d" % (curr_index, len(texts)))
		text_features_vecs[curr_index] = text_to_vec(text, model, num_features)
		curr_index += 1

	return text_features_vecs


# Main program
train_list = []
test_list = []
word2vec_input = []
pred = []

train_data = pd.read_csv('datasets/train.csv')
test_data = pd.read_csv('datasets/test.csv')

# if vector_type == "Word2vec"


# Extracts words from text
if vector_type == "Word2vec" or vector_type == "Word2vec_pretained":
	for i in range(0, len(train_data.text)):

		if vector_type == "Word2vec":
			word2Vec_input.extend(text_to_doublelist(
				train_data.text[i].decode("utf-8"), tokenizer))

		train_list.append(clean_text(train_data.text[i], output_format="list"))
		if i%1000 == 0:
			print("cleaning training text", i)

	for i in range(0, len(test_data.text)):
		test_list.append(clean_text(test_data.text[i], output_format = "list"))
		if i%1000 == 0:
			print("cleaning test text", i)



elif vector_type != "no":
 	for i in range(0, len(train_data.text)):
 		# append raw texts rather than lists as 
 		#Count/TFIDF vectorizers take raw text as inputs
 		train_list.append(clean_text(train_data.text[i]))
 		if i%1000 == 0:
 			print("cleaning training text", i)

 	




# generate vectors from words
if vector_type == "Word2vec_pretrained" or vector_type == "Word2vec":
	if vector_type == "Word2vec_pretrained":
		print("loading the pre-trained model")
		if model_type == "bin":
			model = word2vec.Word2vec.load_word2vec_format(model_name, binary = True)
		else:
			model = word2vec.Word2vec.laod(model_name)

	if vector_type == "Word2vec":
		print("Training word2vec word vectors")
		model = word2vec.Word2Vec(word2vec_input, workers = num_workers,
			size = num_features, min_count = min_word_count,
			window = context, sample = downsampling)

		# if no further training and only query is needed, this trims unnecessary memory
		model.init_sims(replace = True)
		# save the model for later use
		model.save(model_name)

	print("Vectorizing training review")
	train_vec = gen_text_vecs(train_list, model, num_features)
	print("Vectorizing test review")
	test_vec = gen_text_vecs(test_list, model, num_features)

elif vector_type != "no":
	if vector_type == "TFIDF":
		# Unit of gram is "word", only top 5000/10000 words are extracted
		count_vec = TfidfVectorizer(analyzer = "word",
			max_features = 10000,
			ngram_range = (1,2),
			sublinear_tf = True)
	elif vector_type == "Binary" or vector_type == "Int":
		count_vec = CountVectorizer(analyzer = "word",
			max_features = 10000,
			ngram_range = (1,2))

	# return a scipy sparse term-document matrix
	print("Vectorizing input texts")
	train_vec = count_vec.fit_transform(train_list)
	test_vec = count_vec.transform(test_list)


# Dimension Reduction
if dim_reduce == "SVD":
	print("performing dimension reduction")
	svd = TruncatedSVD(n_components = num_dim)
	train_vec = svd.fit_transform(train_vec)
	test_vec = svd.transform(test_vec)
	print("Explained variance ratio =", svd.explained_variance_ratio_.sum())

elif dim_reduce == "chi2":
	print("performing feature selection based on chi2 independce test")
	fselect = SelectKBest(chi2, k=num_dim)
	train_vec = fselect.fit_transform(train_vec, train_data.sentiment)
	test_vec = fselect.transform(test_vec)

# Transform into numpy arrays
if "numpy.ndarray" not in str(type(train_vec)):
	train_vec = train_vec.toarray()
	test_vec = test_vec.toarray()




# Feature Scaling
if scaling != "no":
	if scaler == "standard":
		scaler = preprocessing.StandardScaler()
	else:
		if scaling == "unsigned":
			scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
		elif scaling == "signed":
			scaler = preprocessing.MinMaxScaler(feature_range = (-1,1))

		print("Scaling vectors")
		train_vec = scaler.fit_transform(train_vec)
		test_vec = scaler.fit_transform(test_vec)

# Model training
if training_model == 'RF' or training_model == "BT":
	# Initialize the random Forest or bagged tree based the model chosen
	rfc = RFC(n_estimators = 100, oob_score = True,
		max_features = (None if trainging_model == "BT" else "auto"))
	print("Training %s" %("Random Forest" if training_model == "RF" else "bagged tree"))
	rfc = rfc.fit(train_vec, train_data.sentiment)
	print("OOB Score = ", rfc.oob_score)
	pred = rfc.predict(test_vec)
elif training_model == "NB":
	nb = naive_bayes.MultinomialNB()
	cv_score = cross_val_score(nb, train_vec, train_data.sentiment, cv = 10)
	print("Training Naive Bayes")
	print("cv score = ", cv_score.mean())
	nb = nb.fit(train_vec, train_data.sentiment)
	pred = nb.predict(test_vec)

elif training_model == 'SVM':
	svc = svm.LinearSVC()
	param = {'C': [1e15,1e13,1e11,1e9,1e7,1e5,1e3,1e1,1e-1,1e-3,1e-5]}
	print("Training SVM")
	svc = GridSearchCV(svc, param, cv = 10)
	svc = svc.fit(train_vec, train_data.sentiment)
	pred = svc.predict(test_vec)
	print("Optimized parameters: ", svc.best_estimator_)
	print("Best CV score: ", svc.best_score_)




#output the results
if write_to_csv:
	output = pd.DataFrame(data = {"unique_hash":test_data.unique_hash, 
		"sentiment":pred})
	output.to_csv("submission.csv", index = False)
