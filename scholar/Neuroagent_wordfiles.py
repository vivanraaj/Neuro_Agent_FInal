# this files deals with all things related to the word vectors
import word2vec
from keras import backend as K
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras_helper import NNWeightHelper
import numpy as np
import math

class Vector():
	def __init__(self):
		self.number_of_results = 10
		self.number_analogy_results = 20
		self.autoAddTags = True
		# specify the folder of the pretrained word embeddings
		word2vec_bin_loc = 'scholar/postagged_wikipedia_for_word2vec.bin'
		self.model = word2vec.load(word2vec_bin_loc)
		# This is a list of the tags as organized in the text file
		self.tag_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
		self.load_tags()
		# init the neural network for neuroevolution
		self.network = Sequential()
		self.network.add(Dense(100,input_shape=(100,),activation = 'tanh'))
		
		# below for debug
		#print(self.network.summary())

		
		self.nnw = NNWeightHelper(self.network)
		self.weights = self.nnw.get_weights()
		# keeps track of how many times self.get_results_for_words are called
		self.counter = 0
		#keep track of words it has seen
		self.words_tags_last_seen = {}

	def load_tags(self):
		"""	
		function to load the POS tag counts into a dictionary
		"""
		tag_distribution_loc = 'scholar/postag_distributions_for_scholar.txt'
		# Loads the part of speech tag counts into a dictionary (words to tag string delimited by '-'s)
		with open(tag_distribution_loc) as f:
			word_tag_dist = f.read()

		# Save each word to a list of tags in a global dictionary
		self.word_to_tags = {}
		for line in word_tag_dist.split():
			pieces = line.split('.')
			word = pieces[0]
			tag_counts = pieces[1].split('-')
			tag_counts_as_ints = [int(tag_count) for tag_count in tag_counts]
			self.word_to_tags[word] = tag_counts_as_ints

	def get_verbs(self, noun, snes_weights,tags, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'VB', 'scholar/canon_verbs.txt', False,snes_weights,tags, number_of_user_results)

	def get_canonical_results_for_nouns(self, noun, query_tag, canonical_tag_filename, plural,snes_weights,tags, number_of_user_results):
		if self.autoAddTags:
			noun += '_NNS' if plural else '_NN'
		canonical_pairs = open(canonical_tag_filename)
		result_map = {}
		# For every line in the file of canonical pairs...
		for line in canonical_pairs:
			# ...split into separate words...
			words = line.split()
			if plural:
				if query_tag == 'VB' or query_tag == 'JJ':
					query_string = words[0] + '_' + query_tag + ' -' + words[1] + '_NNS ' + noun
			else:
				if query_tag == 'VB' or query_tag == 'JJ':
					query_string = words[0] + '_' + query_tag + ' -' + words[1] + '_NN ' + noun

			# ...performs an analogy using the words...
			try:
				result_list = self.analogy(query_string,snes_weights,tags)
			except:
				result_list = []
			# ...and adds those results to a map (sorting depending on popularity, Poll method)
			for result in result_list:
				if result in result_map.keys():
					result_map[result] += 1
				else:
					result_map[result] = 1
		final_results = []
		current_max = number_of_user_results
		# While we haven't reached the requested number of results and the number of possible matches is within reason...
		while len(final_results) < number_of_user_results and current_max > 0:
			# ...for every key in the results...
			for key in result_map.keys():
				# ...if the number of times a result has been seen equals the current 'number of matches'...
				if result_map[key] == current_max:
					# ...add it to the list. (This is so that the results are sorted to the list in order of popularity)
					final_results.append(key)
			current_max -= 1
		if len(final_results) >= number_of_user_results:
			return final_results[0:number_of_user_results]
		return final_results

	def analogy(self, words_string,snes_weights,tags):
		"""	
		Return the analogy results for a list of words (input: "king -man woman")
		"""
		positives, negatives = self.get_positives_and_negatives(words_string.split())
		return self.get_results_for_words(positives, negatives,snes_weights,tags)

	def get_positives_and_negatives(self, words):
		positives = []
		negatives = []
		for x in range(len(words)):
			word_arg = words[x]
			if word_arg.startswith('-'):
				negatives.append(word_arg[1:])
			else:
				positives.append(word_arg)
		return positives, negatives

	def get_results_for_words(self, positives, negatives,snes_weights,tags):
		"""	
		Returns the results of entering a list of positive and negative words into word2vec
		"""

		# for first 14 times, we dont use fine-tune the embeddings but use the original pretrained word vector
		# why 14? 15 is the number of verb combination in canon_verbs.txt
		if self.counter > 14:
			# run the function below everytime get_results_for_words gets called
			self.transform_word_vectors(snes_weights,tags)

		indexes, metrics = self.model.analogy(pos=positives, neg=negatives, n=self.number_analogy_results)
		results = self.model.generate_response(indexes, metrics).tolist()
		self.counter += 1

		return self.format_output(results)

	def format_output(self, output):
		"""	
		Changes the output from a list of tuples (u'man', 0.816015154188), ... to a list of single words
		"""
		words = []
		for word_value in output:
			words.append(str(word_value[0]))
		return words


	def return_weights(self):
		return self.weights

	def return_words_tags_last_seen(self):
		return self.words_tags_last_seen

	def return_trained_word2vec(self):
		"""	
		save the models seen in the game
		"""
		labels = []
		tokens = []

		for y in self.words_tags_last_seen:
			new_token = self.model.get_vector(y)						
			tokens.append(new_token)
			labels.append(y)

		return tokens,labels

	def nnw_set_weights(self,weights):
		self.nnw.set_weights(weights)

	def transform_word_vectors(self,snes_weights=None,tags=None):
		"""	
		pass the previous vectors for the word to the neuroevolution algorithm
		and send the new vectors back to the word2vec file
		"""
		self.nnw.set_weights(snes_weights)

		sentence_sequences = []
		sentence_word_vectors = []

		# for debug
		# keep track of errors
		# error = 0
		# error_list = []

		# receive the words from tags code
		for words in tags:
			try:
				x =(words[0].lower()+'_'+words[1])
				# get the vector of the words first 
				# then get the index of the word
				# as the word might not have the vectors.
				# below we get the vectors of the words
	
				sentence_word_vectors.append(self.model.get_vector(x))
				if x in self.words_tags_last_seen:
					index = self.words_tags_last_seen[x]
				else:
					self.words_tags_last_seen[x] = self.model.ix(x)

				# below we get the index of the words
				sentence_sequences.append(index)

			except:
				# error += 1
				# debug to see which word is not on the list
				# error_list.append(x)
				pass

		# convert from list to array

		changed_word2vec_vectors = self.network.predict(np.array(sentence_word_vectors))

		i = 0

		for index in sentence_sequences:
			self.model.vectors[index] = changed_word2vec_vectors[i]

			# check if the vectors changed?
			#word_vectors_changed = self.model.vectors[index]
			i += 1