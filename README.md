# Nimbus-Level-1
Here  is the way I've made the first exercise.

In order to make everything work a lit bit smarter, i've choose to do the thing a little bit different.

First of all, it does not use RAG, but a Neural Network of Reinforcement Learning from Human Feedback. 

The idea is simples, from the web i catch the biggest amount of information from the company CloudWalk, the seppareted in .txt files for each subject ( located in \cloudwalk_docs) this way I can update the informations every time i want. 

From then, created a scrippt named 'training.py' for the creation of the neural network and the begin of the training (as the name say), the training is based on 500 epochs and the evolution graph is show on the file 'curva_aprendizado.png', this script then generate a file named 'cloudwalk_brain.pth' with all the hyperparametters needed to the chatbot 'work'. 

The next step is to create a UI easy to view and customizable, i choose the STREAMLIT just for having more experience with it. There you can type the question and read the Output from the training, but, the learning don't stop there, there was two button, one for positive outputs and other for negative output, every feedback make the connection more straight or weak, deppends if is positive or negative. 

By the end, there's a final script named 'neural_dashboard.py' also using STREAMLIT but this time is just to show the graphs of learning, the ammount of words learned, the x-ray of the connections for the entry and for the exit show by the color blue (for weakness).
