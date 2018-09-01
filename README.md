# Neuroagent : An AI agent that plays any parser text-based games.

### Prerequisites

You will need Anaconda to run the environment. To create the anaconda environment, run the command '$ conda create --name <env> --file <req.txt>'.


### Installing


Clone the repo and download external files at https://drive.google.com/drive/folders/1PClbgG4mgZ6KYxsg7b_khJE9Zto1Hfsf?usp=sharing

and move the files as below:

put in scholar folder:
1) postagged_wikipedia_for_word2vec.bin
2) postag_distributions_for_scholar.txt
3) canon_verbs.txt

put in agents folder:
1) master_verbs.p



## Usage

To run the training agent, go to the home folder directory and type 'python main.py 'game_name' 'run_no' 'population size' 'generations number' to start training of the agent.


To evaluate the results, run the files below:

1) analyze_pickle_file_plot_graph.ipynb
2) analyze__tsne.ipynb


## Acknowledgments

* Hat tip to anyone whose code was used
