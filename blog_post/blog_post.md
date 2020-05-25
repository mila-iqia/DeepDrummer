# DeepDrummer

DeepDrummer is a drum loop generation tool that uses active learning to learn the preferences (or current artistic intentions) of a human user from a small number of interactions. The principal goal of this tool is to enable an efficient exploration of new musical ideas. We train a deep neural network classifier on audio data and show how it can be used as the core component of a system that generates drum loops based on few prior beliefs as to how these loops should be structured.

DeepDrummer is composed of three main components:
- the web interface (presenting the user with drum loops to be rated as either *like* or *dislike*)
- the critic (learning what the user likes, serving as filter for future generated loops)
- the generator (propose new drum loops to achieve a good score with critic)

The critic network takes a drum loop's raw audio preprocessed as MFCC features, and then applies successive layers of convolutions in order to finally output its prediction about the probability that the user will *like* the drum loop.

[Maybe put a sketch here with the 3 components? The kind of thing you wouldn't put in a scientific paper, but would ?put on a web site to keep people entertained while reading.]

The generator is a function that outputs random grid sequencer patterns with 16 time steps during which 4 randomly-selected drum sounds can be triggered.
We choose a very basic generator that does not have any trainable parameters, and constitutes a source of patterns that has few priors on musical structure.

Combined together, the feedback from the critic can serve as a powerful filter for the output of the generator. As a result, the interface will present only the most relevant drum loops to the user for rating.

We show a demonstration of all the pieces working together in the following YouTube video.

EMBED YOUTUBE VIDEO HERE.

[Should we explain the novelty in more explicit terms?]

# Experiment

We ran an experiment with 25 participants in which we


Explain the experiment without spending too much time on details.

Show the figures with the results.

Show best/worse samples from participants, and some of our own favorites.

# Dataset released

Briefly describe the dataset that we're publishing, and give the download link. Make it sound like it's a contribution in its own right, but don't sell it too hard because we don't want to eclipse the actual core DeepDrummer.

# Running the code

Reproducing the experiment.

Starting from pre-trained model.

# Links

Reminder about links for arXiv, github, youtube and dataset, even though the links were given in the previous sections.

bibtex to cite us

3
2
1

Je viens de penser à quelque chose. Je regardais les folders dans ISMIR2020 et j'ai constaté qu'on avait encore un setup avec "deepgroove" dans la structure.

The DeepDrummer experiment was run on a computer with a INSERTBRAND GPU, which is way more computational power than necessary.
