# AI vs Original Image Classification

This project focuses on building a machine learning model to classify whether an image is *AI-generated* or *real (original)*. The model uses convolutional neural networks (CNNs) to extract visual features and make predictions.

---

## 🧠 Problem Statement

With the rise of powerful generative AI tools (like DALL·E, Midjourney, and Stable Diffusion), it has become difficult to visually distinguish AI-generated images from original human-taken photographs. This project aims to tackle this challenge by training a model that can classify images as *AI-generated* or *Real*.

---

## 📁 Dataset

The dataset consists of:
- AI-generated images (created using tools such as DALL·E, Midjourney, or Stable Diffusion)
- Real images (collected from open datasets or photography sources)

> 📌 *Note*: Dataset is not included in this repository due to size and copyright limitations. Please contact me if you need a sample dataset or reference.

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy & Pandas
- Matplotlib & Seaborn
- Jupyter Notebook

---

## 🏗️ Model Architecture

- Convolutional Neural Network (CNN)
  - Conv2D → MaxPooling → Dropout → Dense
- Activation functions: ReLU and Sigmoid
- Binary classification output: *AI (0)* or *Real (1)*

---

## 📊 Model Performance

- Accuracy: ~92% (based on sample data)
- Precision & Recall
- Confusion Matrix
- ROC-AUC Score

> 🖼️ Example Output:  
> Image #101: AI-generated (Confidence: 95%)  
> Image #102: Real (Confidence: 97%)

---

## 🚀 Future Enhancements

- Use pre-trained models (ResNet, EfficientNet) for better accuracy
- Expand dataset with more diverse sources
- Deploy as a Streamlit or Flask web app

---

## 🙋 About the Author

*👤 Name*: Varra Guruswamy  
*🎓 Education*: Bachelor of Science in Computer Science (Graduated 2024)  
*💼 Internships*:  
- AI/ML Internship  
- Web Development Internship  

*🛠️ Skills*:
- Python, Machine Learning, Deep Learning (CNNs)  
- Data Analysis & Visualization  
- GitHub & Jupyter Notebooks  
- Streamlit (basic)  

*📫 Contact*:  
Email: varraguruswamy@gmail.com

---

## 📄 Disclaimer

This project is for learning and demonstration purposes. Datasets used belong to their respective creators or public sources
