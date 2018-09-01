# this file to run the agent
import scholar.Neuroagent_wordfiles as vw
import verbFinder
import nltk
import re
import random as rand

class NeuroAgent():
	def  __init__(self):	
		# load the verbs
		self.manipulation_list = ['throw', 'spray', 'stab', 'slay', 'open', 'pierce', 'thrust', 'exorcise', 'place', 'jump', 'take', 'make', 'read', 'strangle', 'swallow', 'slide', 'wave', 'look', 'dig', 'pull', 'put', 'rub', 'fight', 'ask', 'score', 'apply', 'take', 'knock', 'block', 'kick', 'step', 'break', 'wind', 'blow', 'crack', 'drop', 'blast', 'leave', 'yell', 'skip', 'stare', 'hurl', 'hit', 'kill', 'glass', 'engrave', 'bottle', 'pour', 'feed', 'hatch', 'swim', 'spray', 'melt', 'cross', 'insert', 'lean', 'sit', 'move', 'fasten', 'play', 'drink', 'climb', 'walk', 'consume', 'kiss', 'startle', 'shout', 'close', 'cast', 'set', 'drive', 'lift', 'strike', 'startle', 'catch', 'board', 'speak', 'think', 'get', 'answer', 'tell', 'feel', 'get', 'turn', 'listen', 'read', 'watch', 'wash', 'purchase', 'do', 'sleep', 'fasten', 'drag', 'swing', 'empty', 'switch', 'slip', 'twist', 'shoot', 'slice', 'read', 'burn', 'hop', 'rub', 'ring', 'swipe', 'display', 'scrub', 'hug', 'operate', 'touch', 'sit', 'sweep', 'fix', 'walk', 'crack', 'skip']
		self.manipulation_list += ['wait', 'point', 'light', 'unlight', 'use', 'ignite', 'wear', 'remove', 'unlock', 'lock', 'examine', 'inventory', '']
		# the games recognizes these movement commands -> https://gamefaqs.gamespot.com/pc/564446-zork-i/faqs/20848
		self.navigation_list = ['north', 'south', 'west', 'east', 'northwest', 'southwest', 'northeast', 'southeast', 'up', 'down', 'enter', 'exit', 'drop','in','out']
		self.verb_list = self.manipulation_list + self.navigation_list
	
		undesirable_verb = ['save','quit','restart']
		for v in undesirable_verb:
			if v in self.verb_list:
				self.verb_list.remove(v)		

		self.VPD = {}
		self.preposition_list = ['with', 'in', 'at', 'above', 'under']

		print("Loading word vectors")
		# initialize the Neuroagnet_wordfiles class
		self.vw = vw.Vector()

		self.verbFinder = verbFinder.verbFinder()
		print("Loading Verbs")
		self.verbFinder.addVerbsFromFile("agents/master_verbs.p")
		for v in self.verb_list:
			words = self.verbFinder.wordsForVerb(v)
			preps = []
			if len(words.keys()) > 0:
				for w in list(words.keys()):
					if w in self.preposition_list:
						preps.append(w)
			else:
				preps = ''
	 	
			if len(preps) == 0:
				preps = self.preposition_list

			self.VPD[v] = preps

		self.last_state = ''
		self.current_state = ''
		self.last_narrative = ''
		self.current_narrative = ''
		self.last_verb = ''
		self.last_object = ''
		self.last_action = 'look'
		self.inventory_list = []
		self.inventory_text = ""
		self.TWO_WORD_OBJECTS = True
		self.inventory_count = 0
		self.look_flag = 0
		self.get_flag = 0
		self.packrat_count = 0 #am I just getting all all the time? (because the game narrative is too variable)
		self.inventory_count = 0 #am I just checking inventory? (because the game narrative is too variable)
		self.game_steps = 0
		self.exploration_counts = {}
		self.visited_states = []
		self.visited_narratives = []
		self.verbs_for_noun = {}
		self.alreadyTried = {}
		self.success = {}
		self.total_points_earned = 0
		# for debugging
		# self.matching_evaluate = []
		# self.track_generation_no = 0

	def update(self, reward):
		"""	
		function to update total_points_earned
		"""
		self.total_points_earned += reward

	def get_total_points_earned(self):
		"""	
		function to return total points earned
		"""
		return self.total_points_earned

	def agent_return_weights(self):
		"""	
		function to return the weights of the neural network structure
		"""
		return self.vw.return_weights()

	def agent_return_word_seen(self):
		"""	
		function to return the dictionary that contains the words seen in the game state and its index number
		"""
		return self.vw.return_words_tags_last_seen()

	def agent_return_models(self): 
		"""	
		function to return the words and its correspending word vectors
		"""
		return self.vw.return_trained_word2vec()

	def take_action(self, narrative, snes_weights):
		"""	
		function to return the action command

		Args:
				narrative (str): game state text description
				snes weights (arr): weights of the neural network
		"""
			
		self.game_steps += 1

		#every 1000 steps, reset the alreadyTried list
		#(this helps the agent try new things and keeps it from
		#gettibg 'stuck in a rut'. It also helps compensate for
		#unobservable state changes.)
		if self.game_steps%1000 == 0:
			for state in self.alreadyTried.keys():
				for obj in self.alreadyTried[state].keys():
					for vrb in self.alreadyTried[state][obj].keys():
						self.alreadyTried[state][obj][vrb] = 0

		#process results of look and inventory commands
		if self.last_action == "inventory":
			self.inventory_list = self.find_objects(narrative)
			self.inventory_text = narrative
			self.get_flag = 0
		elif self.last_action == "look":
			self.last_state = self.current_state
			self.last_narrative = self.current_narrative
			self.current_narrative = re.sub(r'\d+', '', narrative)
			#state is the narrative plus inventory
			self.current_state = re.sub(r'\d+', '', narrative + self.inventory_text)


		#execute 'look' command every other step.
		#(This helps to make the state space more observable)
		if self.look_flag == 1:
			self.look_flag = 0
			self.last_action = "look"
			return "look"
		else:
			self.look_flag = 1

		#check inventory whenever we execute a 'get' command.
		#(inventory results are included as part of the state space)
		if self.last_verb == 'get':
			self.get_flag = 1
		if self.get_flag > 0 and self.inventory_count < 5:
			self.last_action = "inventory"
			self.last_verb = "inventory"
			self.inventory_count += 1
			self.get_flag = 0
			return "inventory"
	
		#try 'get all' in each new state	
		if self.current_narrative not in self.visited_narratives and self.packrat_count < 10:
			self.visited_narratives.append(self.current_narrative)
			self.last_action = 'get all'
			self.last_verb = 'get'
			self.last_object = 'all'
			self.get_flag = 1
			self.packrat_count += 1
			if self.packrat_count > 5: 
				# why add to 100??
				self.packrat_count = 100
		else:			
			#select an action
			self.last_action = self.chooseAction(self.current_state,snes_weights)
			print ("Action is " + self.last_action.strip()) 
		
		return self.last_action.strip()



	def find_objects(self, narrative):
		"""	
		function to return the nouns in the game state text descriptionn
		"""

		#Assume an object is manipulatable if it appears as a noun in the game text
		# each time this is run, we set the tags to None		
		tokens = nltk.word_tokenize(narrative)
		self.tags = nltk.pos_tag(tokens)
		# extract the all words of nouns and their tags
		nouns = [word for word,pos in self.tags if word.isalnum() and (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]

		# for words that has adjective and nouns like white house.	
		# if previous word is adjective and next word is noun
		for i in range(0, len(self.tags) - 1):
				if (self.tags[i][1] == "JJ") and (self.tags[i+1][1] in ["NN", "NNP", "NNS", "NNPS"]):
					nouns.append(self.tags[i][0] + " " + self.tags[i+1][0])

		return nouns


	def chooseAction(self, game_text, snes_weights):
		"""	
		function to choose the desired action
		"""

		if self.packrat_count > 0:
			self.packrat_count -= 1  # we've done something besides just 'get all'

		if self.inventory_count > 0:
			self.inventory_count -= 1  # we've done something besides just 'get all'

		objects = self.find_objects(game_text) + ['']

		# choose a random object of the game text
		obj = rand.choice(objects)

		# if this is a new state or object, then
		# initialize the proper dictionary keys
		if game_text not in self.alreadyTried.keys():
			self.alreadyTried[game_text] = {}

		if obj not in self.alreadyTried[game_text].keys():
			self.alreadyTried[game_text][obj] = {}
			# trying every single verb in the verb list
			for v in self.verb_list:
				self.alreadyTried[game_text][obj][v] = 0

		if game_text not in self.success.keys():
			self.success[game_text] = {}

		if obj not in self.success[game_text].keys():
			self.success[game_text][obj] = {}
			for v in self.verb_list:
				self.success[game_text][obj][v] = 0

		# Check to see whether the last action was successful
		# If it was, then remember that this was a useful action
		# ('Success' is defined as eliciting a state change.)
		if self.last_state != self.current_state:
			if self.last_state not in self.success.keys():
				self.success[self.last_state] = {}
			if self.last_object not in self.success[self.last_state].keys():
				self.success[self.last_state][self.last_object] = {}
				for v in self.verb_list:
					self.success[self.last_state][self.last_object][v] = 0
			self.success[self.last_state][self.last_object][self.last_verb] = 1

		# choose the next action
		r = rand.randint(0, 10)
		if r == 0 and obj != '':
			# get a verb/preposition/object combo
			commands = self.getCommands(self.getTryList(game_text, obj, snes_weights),
										self.find_objects(game_text) + self.inventory_list)
			action = rand.choice(commands)
			return action
		else:
			# get a verb/object combination
			vrb = self.getVerb(game_text, obj, snes_weights)
			self.alreadyTried[game_text][obj][vrb] = 1
			self.last_verb = vrb
			self.last_object = obj
			return vrb + ' ' + obj

	def getVerb(self, game_text, input_object, snes_weights):
		"""	
		#returns a verb that:
		# (A) satisfies the active search criterion
		# (B) is in the agent's verb_list
		# (C) has not already been tried in this state with this object	
		"""

		tryList = self.getTryList(game_text, input_object, snes_weights)	
		vrb = rand.choice(tryList)

		return vrb
		

	def getCommands(self):

		"""	
		using the dictionary, return a list of commands
		so like saying get house with room
		"""
		sents = []
		#Verb
		for v in self.verb_list:
				#Noun
				for obj in self.object_list:
						#Dictionary of prepositions according to verbs
						for key in self.VPD.keys():
								#set or list of prepositions
								for prep in self.VPD[key]:
										#second Noun
										for obj2 in self.object_list:
												sentence = "{} {} {} {}". format(v, obj, prep, obj2)
												sents.append(sentence)

		return sents


	#using the dictionary, return a list of commands
	def getCommands(self,verbs,objects):
		"""	
		using the dictionary, return a list of commands
		so like saying get house with room
		"""

		if '' in verbs:
				verbs.remove('')
		if '' in objects:
				objects.remove('')

		sents = []

		for v in verbs:
				#Noun
				for obj in objects:
						#Dictionary of prepositions according to verbs
						for key in self.VPD.keys():
								#set or list of prepositions
								for prep in self.VPD[key]:
										#second Noun
										for obj2 in objects:
												sentence = "{} {} {} {}". format(v, obj, prep, obj2)
												sents.append(sentence)

		return sents


	def getTryList(self, game_text, input_object,snes_weights):
		"""	
		some objects are composed of two words (usually an adjective and an object)
		If that is the case, then consider only the second word out of the pair
		"""

		obj = input_object
		if len(input_object.split()) > 1:
			obj = input_object.split()[-1]

		obj = obj.lower()

		#identify a set of verbs that seems to 'match' the current object of interest.
		#(This is accomplished using one of three different methods, all of which
		#rely on the Wikipedia corpus for the extraction of common-sense knowledge
		#about the relationship of verbs to specific objects.)

		if obj in self.verbs_for_noun.keys():
			# check from own list of verbs first
			matching_verbs = self.verbs_for_noun[obj]
		else:
			matching_verbs = self.vw.get_verbs(obj,snes_weights,self.tags,30)
			for i in range(len(matching_verbs)):
				matching_verbs[i] = matching_verbs[i][:-3]
			self.verbs_for_noun[obj] = matching_verbs

		# for debugging
		#if self.track_generation_no in [100]:
		#	evaluate = 'wooden'
		#	self.matching_evaluate = self.vw.get_verbs(evaluate,snes_weights,self.tags,30)
			

		tryList = []

		#we first try to manipulate the objects extracted from the game text,
		#so we look for the intersection between our manipulation list and the
		#wikipedia verbs that match this object
		for v in matching_verbs:
			if v in self.manipulation_list:
				if self.alreadyTried[game_text][input_object][v] == 0:
					tryList.append(v)

		#certain verbs are so useful that we ALWAYS include them in the try list
		if 'open' not in tryList and 'open' in self.alreadyTried[game_text][input_object].keys() and self.alreadyTried[game_text][input_object]['open'] == 0:
			tryList.append('open')
		if 'get' not in tryList and 'get' in self.alreadyTried[game_text][input_object].keys() and self.alreadyTried[game_text][input_object]['get'] == 0:
			tryList.append('get')
		if 'put' not in tryList and 'put' in self.alreadyTried[game_text][input_object].keys() and self.alreadyTried[game_text][input_object]['put'] == 0:
			tryList.append('put')

		#if we've tried everything we can think of, then we proceed
		#to try either (A) Things that worked, or (B) navigate elsewhere
		if len(tryList) == 0:
			if obj not in self.success[game_text].keys():
				self.success[game_text][obj] = {}
				for v in self.verb_list:
					self.success[game_text][obj][v] = 0.0
			tryList = list(self.success[game_text][obj].keys()) + self.navigation_list

		#if nothing seems to be working, then navigate away
		if len(tryList) == 0:
			tryList = self.navigation_list

		return tryList

	def pass_snes_centre_weight(self,snes_centre_weights):
		"""	
		function to pass the new weights of the neural network to the neural network structure
		"""
		self.vw.nnw_set_weights(snes_centre_weights)