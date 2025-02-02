# A3-Make-Your-Own-Machine-Translation-Language
## st124952 Patsachon Pattakulpong

Thai - English Translation 

## 1. Source
- My word dictionary that i use is SCB_MT_EN_TH v1.0 from SCB (which provide in the folder named as data)
- FYI: my model .pt cannot upload cause it too large so you can access via this link: https://drive.google.com/drive/folders/1G1Z4PoNgbeBK5TF3vhaSXamxfSps7Rsx?usp=sharing

## 2. Dataset Preparation Process
Start by importing the parallel corpus from two separate files containing English (en.txt) and Thai (th.txt) text data. Leverage TorchData's capabilities to efficiently handle and process these language pairs.
Before moving forward with model training, the text requires careful preprocessing. Clean each sentence by removing excess whitespace and standardizing the format. This normalization step ensures consistent data quality across both languages.
Once the data is cleaned, divide the complete dataset using a 70-20-10 split ratio. This creates three distinct sets: a training set comprising 70% of the data, a validation set with 20%, and the remaining 10% forming the test set.
The next crucial phase involves tokenization, which differs for each language. 
For English text, implement SpaCy's tokenization tools to break sentences into individual words. Thai text presents a unique challenge since it doesn't use spaces between words. 
Address this by employing PyThaiNLP's specialized word segmentation algorithms.
Finally, transform the tokenized text into numerical format. This conversion is essential as neural networks can only process numerical data. 
Each token gets mapped to a unique integer value, creating sequences that the model can understand and process.
This systematic approach ensures your dataset is properly structured and ready for model training.

## 3. Attention Mechanism Comparison Analysis and Evaluation

| Attentions | Training Loss | Traning PPL | Validation Loss | Validation PPL |
|-------------------|-------------|---------------|---------------|--------------------|
| General Attention          | 3.173     | 23.884     | 5.427    | 227.555         |
| Multiplicative Attention    | 3.172     | 23.854       | 23.854       | 233.574            |
| Additive Attention             | 3.113    | 22.478      | 5.824     | 338.165            |

| Attentions | BLEU Score (Accuracy) | compu-tational efficiency | Perplexity (PPL) |
|-------------------|-------------|---------------|---------------|
| General Attention          | 0.000     | 15m 1s       | 167.28067814348537    |
| Multiplicative Attention    | 0.000     | 16m 3s       | 178.29528682922708      |
| Additive Attention             | 0.000     | 35m 2s       | 165.58213232068604      |


## Plot Loss-Val Curve and heat map for General Attention
![image](https://github.com/user-attachments/assets/b1192179-aace-4cbc-bc2e-046bffc08435)
![image](https://github.com/user-attachments/assets/b7cbe174-0a2e-4eea-84db-98a61993e77b)
### 1. Training and Validation Loss Curve (General Attention)
The first image shows the training and validation loss curves for the general attention mechanism during training. The **train loss** (blue) consistently decreases, indicating the model is learning from the data. 
The **validation loss** (orange) initially decreases but then slightly increases, which may suggest some overfitting as training progresses.

### 2. Attention Heatmap  
The second image is a heatmap representing the attention weights of the model during translation. The x-axis represents the input English sentence tokens, while the y-axis represents the generated Thai tokens. 
Darker areas indicate higher attention weights, meaning the model focuses more on those specific words during translation. This visualization helps interpret how the model aligns source and target words.


## Plot Loss Curve and heat map of Multiplicative Attention
![image](https://github.com/user-attachments/assets/dcf506a8-8407-4d7c-a5fe-3cdbeb823128)
![image](https://github.com/user-attachments/assets/64c00b17-fe9d-42f1-bd94-737ec4f0a41f)
### 1. Training and Validation Loss Curve (Multiplicative Attention)  
The first image presents the loss curves for the multiplicative attention model. 
The **train loss** (blue) consistently decreases, indicating the model is learning effectively. However, 
the **validation loss** (orange) flattens and slightly increases after a few updates, possibly suggesting overfitting.

### 2. Attention Heatmap (Multiplicative Attention)  
The second image visualizes the attention weights of the multiplicative attention mechanism. 
The x-axis represents input English tokens, while the y-axis represents generated Thai tokens. Darker regions indicate stronger attention focus. 
This heatmap helps interpret how the model aligns words between source and target sentences during translation.

## Plot Loss Curve and heat map of Additive Attention
![image](https://github.com/user-attachments/assets/b209f79f-81b9-4ed8-bdfb-52b4cfa32753)
![image](https://github.com/user-attachments/assets/523f827f-267d-4478-b4e4-30d0467b96ab)
### 1. Training and Validation Loss Loss Curves
The training shows consistent improvement over 10 updates, with the **training loss** (blue line) steadily decreasing from ~6.5 to ~3.0. 
The validation loss (orange line) initially decreases but starts to increase slightly after update 4, suggesting potential overfitting in later stages.
### 2. Attention Heatmap
The attention heatmap visualizes the model's focus when translating between English and Thai. 
Darker cells indicate stronger attention weights between source and target tokens. The visualization includes special tokens (<sos>, <eos>, <unk>) and 
shows how the model learns to align words between the two languages during translation.
These visualizations demonstrate both the model's learning progress and its attention patterns during translation tasks.

## 4.Analysis and Selection of Optimal Attention Mechanism
On this assignment i choose additive attention because from Perplexity score during evaluation stage,  
in my opinion agree with a lower perplexity score is considered better cause it indicates that the model is more confident in its predictions and has a better understanding of the language patterns

## Application 
How to run: python app.py


Uploading Screen Recording 2568-02-03 at 5.27.32 AM.mov…

<img width="1440" alt="Screenshot 2568-02-03 at 5 28 45 AM" src="https://github.com/user-attachments/assets/8c167446-fd47-463e-a1e0-ff0d6371d9cf" />

