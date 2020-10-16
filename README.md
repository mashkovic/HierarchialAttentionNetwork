# Code for HAN experiment - Mashrur

DISCLAIMER: This is not a very well written documentation but is definitely better than you-know-who :)

<br>

## Declaration :
<hr>
Majority of the code is taken from https://github.com/TovlyDeutsch/Linguistic-Features-for-Readability 

The model described in the associated research paper and the code has been tweaked to achieve our purposes.  

<br>

## Results and Evaluation
<hr>
The results should be contained in <code>out/Evaluation.txt</code>. The model was run with 5 k-folds. Learning Rate: 0.0001, Batch Size : 64. 20 Epochs, with patience of 10 and min_delta of 0.0001.

<br><br>

## How to run the experiment? 
<hr>
Ensure the data is in <code>myCorpus.csv</code> or provide alternate file path in <code>opt</code> dictionary in the <code>main.py</code>. All the configurations for running the model is contained in <code>main.py</code>. For changing the model itself, please modify: <code>HAN.py, word_att_model.py</code> and <code>sent_att_model.py</code>. 