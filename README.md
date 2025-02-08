# TopicDetection-MHRN
### **🚀 README: Multimodal Crisis Detection (CNN + RNN) on CrisisMMD Dataset**  

---

## **📌 Project Overview**  
This project implements a **Multimodal Hierarchical Reasoning Network (MHRN)** to classify crisis-related tweets using **both text (RNN) and images (CNN)**. The model processes **textual and visual features**, fuses them, and predicts whether a tweet is **informative or not** in a crisis situation.  

- **Text Processing:** RNN (BiLSTM/GRU) for feature extraction.  
- **Image Processing:** CNN (ResNet18/ResNet50) for image feature extraction.  
- **Multimodal Fusion:** Combines text and image features for classification.  

---

## **📂 Project Structure**  

```
├── data/                  
│   ├── annotations/       # Raw TSV files with tweet texts & image metadata  
│   ├── data_image/        # Image dataset  
│   ├── train.csv          # Preprocessed training data (text + image + label)  
│   ├── test.csv           # Preprocessed test data  
│  
├── models/                
│   ├── mhrn_model.pth     # Trained model weights  
│  
├── src/                   
│   ├── datasets/          # Dataset processing  
│   │   ├── twitter_dataset.py   # CrisisMMDataset class  
│   │   ├── preprocess.py        # Preprocesses raw TSV files into train/test CSV  
│   ├── models/            # Model architecture  
│   │   ├── mhrn.py        # CNN + RNN architecture  
│   │   ├── utils.py       # Loss function & metric computation  
│   ├── train.py           # Training script  
│   ├── evaluate.py        # Model evaluation  
│  
├── config.py              # Project configurations  
├── README.md              # Project documentation  
```

---

## **⚙️ Installation & Setup**  

### **🔹 1️⃣ Clone the Repository**
```bash
git https://github.com/nimz78/TopicDetection-MHRN
cd TopicDetection-MHRN
```

### **🔹 2️⃣ Create a Virtual Environment & Install Dependencies**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **🔹 3️⃣ Download & Prepare CrisisMMD Dataset**
-⚠️this project already has data, If you want to use all data of CrisisMMD dataset do this part.⚠️
- Download the dataset from [CrisisMMD on Kaggle](https://www.kaggle.com/datasets/mohammadabdulbasit/crisismmd).  
- Place it inside the `data/` directory.  

### **🔹 4️⃣ Preprocess the Dataset**
```bash
python src/datasets/preprocess.py
```
✅ This generates `train.csv` & `test.csv` in `data/`.

---

## **🚀 Training the Model**
```bash
python src/train.py
```
✅ The trained model will be saved in `models/mhrn_model.pth`.

---

## **📊 Evaluating the Model**
```bash
python src/evaluate.py
```
✅ Example Output:
```
Accuracy: 87.5%
Precision: 85.2%
Recall: 88.1%
F1-score: 86.5%
```

---

## **🛠 Model Architecture**  
The **MHRN model** consists of:  
1️⃣ **Text Feature Extraction:**  
   - Embedding Layer → BiLSTM/GRU → Hidden Representation  
2️⃣ **Image Feature Extraction:**  
   - ResNet CNN → Feature Vector  
3️⃣ **Multimodal Feature Fusion:**  
   - Concatenates Text + Image Features  
4️⃣ **Classification:**  
   - Fully Connected (FC) Layer → Softmax Output  

---

## **📌 Expected Input & Output**
### **🔹 Example Input**
| **Tweet Text** | **Image** |
|--------------|---------|
| `"Massive flooding in Houston after Hurricane Harvey."` | 🌊 Flooded street image |
| `"Look at my cat sitting on my laptop!"` | 🐱 Cat image |

### **🔹 Example Output**
| **Tweet Text** | **Image** | **Prediction** |
|--------------|---------|--------------|
| `"Massive flooding in Houston after Hurricane Harvey."` | 🌊 Flooded street | ✅ **Informative** |
| `"Look at my cat sitting on my laptop!"` | 🐱 Cat image | ❌ **Not Informative** |

---

## **📬 Contact & Contribution**
- **Author:** [Nima Zare]  
- **GitHub:** [TopicDetection-MHRN](https://github.com/nimz78/TopicDetection-MHRN)  
- **Issues & Contributions:** Feel free to open issues or submit PRs!  

🚀 **Let me know if you need any modifications!** 🔥🔥🔥