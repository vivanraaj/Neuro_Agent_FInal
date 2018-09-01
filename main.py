import textplayer.textPlayer as tP
import argparse
import pickle
from timeit import default_timer as timer
import sys
import agents.Neuroagent_agent as Neuroagent_agent
from snes import SNES



def save(generations=None,track_scores=None,game=None,run=None):
	"""	
	save the scores at intervals during the game incase it crashes.
	"""
	if generations in [1,49,99,149,199,249,299,349,399]:
		with open('./completed_runs/'+game+'/finalruns/run_'+str(run)+'/track_scores_'+game+'_pop50gen500.pkl', 'wb') as f:
			pickle.dump(track_scores, f)

def save_tokens_labels_words_seen(generations=None,tokens=None,labels=None,game=None,run=None):	
	"""	
	save the words seen dictionary at intervals during the game incase it crashes
	"""
	if generations in [1,100,200,300,400]:
		path = './completed_runs/'+game+'/finalruns/run_'+str(run)+'/saved_pickles/test_word_seen_vectors_at_'+str(generations)+'_.pkl'		
		path2 = './completed_runs/'+game+'/finalruns/run_'+str(run)+'/saved_pickles/test_word_seen_vocabs_at_'+str(generations)+'_.pkl'
		with open(path, 'wb') as i:
			pickle.dump(tokens, i)
		with open(path2, 'wb') as k:
			pickle.dump(labels, k)

def train_agent(game_chosen,game,run,pop_size,generations):
	"""	
	function to run the game

	   Args:
			game_chosen (str): name of the game
			game(str): folder name
			run(int): folder name
			pop_size: population size for the SNES algorithm
			generations: number of generations for the SNES algorithm
	"""

	# start the desired game file
	textPlayer = tP.TextPlayer(game_chosen)

	# initialize a list to keep track of the game scores
	track_scores = []

	# set last score to zero
	last_score = 0

	# initialize the agent
	agent = Neuroagent_agent.NeuroAgent()

	# return the original weights of the neural network structure in the neuroevolution
	initial_weights = agent.agent_return_weights()

	state = textPlayer.run()
	last_state = state

	# pass variables to SNES
	snes = SNES(initial_weights, 1, pop_size)

	# start the timer
	start = timer()


	# iterate through number of generations
	for i in range(0, generations):
		asked = snes.ask()
		told = []
		j = 0
		for asked_j in asked:
			# use SNES to set the weights of the NN
			last_state = state
			action = agent.take_action(state,asked_j)
			state = textPlayer.execute_command(action)
			print('{0} >>> {1}'.format(action,state))
			print('This is Population No. {0} of Generation no. {1}'.format(j,i))
			if textPlayer.get_score() != None:
				score, possible_score = textPlayer.get_score()
				# if previous state is equal to current state, then agent reward gets deducted by value of -0.2 otherwise reward multiplies by a factor of * 1.2
				reward = score - last_score
				if last_state == state:
					agent_reward = reward - 0.2
				else:
					agent_reward = reward * 1.2
				told.append(agent_reward)
				last_score = score
				agent.update(reward)
				accumulated_reward = agent.get_total_points_earned()
				print  ('Your overall score is {0} and you gained reward of {1} in the last action and agent reward of {2}'.format(accumulated_reward,reward,agent_reward))
				track_scores.append((state,j,i,action,agent_reward,reward,accumulated_reward))
			else:
				#in case the game cant retrieve the score from the game engine, set the score for that generation to 0
				score = 0
				# if previous state is equal to current state, then agent reward gets deducted by value of -0.2 otherwise reward multiplies by a factor of * 1.2
				reward = score - last_score
				if last_state == state:
					agent_reward = reward - 0.2
				else:
					agent_reward = reward * 1.2
				told.append(agent_reward)
				last_score = score
				agent.update(reward)
				accumulated_reward = agent.get_total_points_earned()
				print  ('Your overall score is {0} and you gained reward of {1} in the last action and agent reward of {2}'.format(accumulated_reward,reward,agent_reward))
				track_scores.append((state,j,i,action,agent_reward,reward,accumulated_reward))
			j += 1
		snes.tell(asked,told)
		save(i,track_scores,game,run)
		tokens, labels = agent.agent_return_models()
		save_tokens_labels_words_seen(i, tokens, labels,game,run)

	# at end of training
	# pass the final weights to the neural network structure
	snes_centre_weights = snes.center
	agent.pass_snes_centre_weight(snes_centre_weights)

	word_seen = agent.agent_return_word_seen()

	# save the scores
	with open('./completed_runs/'+game+'/finalruns/run_'+str(run)+'/track_scores_'+game+'_pop50gen500.pkl', 'wb') as f:
		pickle.dump(track_scores, f)

	# save the words seen in the game
	with open('./completed_runs/'+game+'/finalruns/run_'+str(run)+'/words_seen_'+game+'_pop50gen500.pkl', 'wb') as h:
		pickle.dump(word_seen, h)
	with open('./completed_runs/'+game+'/finalruns/run_'+str(run)+'/test_word_seen_vectors_'+game+'_pop50gen500.pkl', 'wb') as i:
		pickle.dump(tokens, i)
	with open('./completed_runs/'+game+'/finalruns/run_'+str(run)+'/test_word_seen_vocab_'+game+'_pop50gen500.pkl', 'wb') as k:
		pickle.dump(labels, k)

	# end the timer
	end = timer()
	print('The time taken to run the traning is {0}'.format(end - start))
	textPlayer.quit()


# below for debugging
# sys.argv += 'zork1 6 5 50'


parser = argparse.ArgumentParser(description='Please type in the game name,run no, population size,generation size?')
parser.add_argument('file',help='Specify game name?',type = str)
parser.add_argument('run_no',help='What run no?',type = int)
parser.add_argument('pop_size',help='What is the population number for the SNES algorithm?',type = int)
parser.add_argument('gen_size',help='What is the generation number for the SNES algorithm?',type = int)


# parse arguments
args = parser.parse_args()
game = args.file
game_chosen = game+'.z5'
run = args.run_no
pop_size = args.pop_size
generations = args.gen_size

if __name__ == '__main__':
	train_agent(game_chosen,game,run,pop_size,generations)