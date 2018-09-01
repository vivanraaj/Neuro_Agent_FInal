#######################################################################################
# This set of codes has been adapted from https://github.com/danielricks/BYU-Agent-2016
#######################################################################################

import pickle

class verbFinder:

	def __init__(self):
		self.verbs = {}
		self.nouns = {}
		self.preps = {}

	def addVerbsFromFile(self, otherVerbPickleFilename):
		otherVerbs = pickle.load(open(otherVerbPickleFilename, 'rb'))
		for verb in otherVerbs:
			if verb in self.verbs:
				for word in otherVerbs[verb]:
					if word in self.verbs[verb]:
						self.verbs[verb][word] += otherVerbs[verb][word]
					else:
						self.inc_verbs(verb, word)
			else:
				for word in otherVerbs[verb]:
					self.verbs[verb] = {}
					if word not in self.verbs[verb]:
						self.verbs[verb][word] = 0
					self.verbs[verb][word] += otherVerbs[verb][word]

	def wordsForVerb(self, verb):
	        if verb in self.verbs.keys():
		     return self.verbs[verb]
		return {}

	def inc_verb(self, verb, word):
		if verb not in self.verbs:
			self.verbs[verb] = {}
		if word not in self.verbs[verb]:
			self.verbs[verb][word] = 0
		self.verbs[verb][word] += 1