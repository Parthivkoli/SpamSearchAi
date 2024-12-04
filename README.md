# SpamSearchAi

SpamSearchAi is an advanced email classification tool powered by **Natural Language Processing (NLP)** and **Machine Learning**. Designed to protect your inbox, SpamSearchAi provides accurate email filtering to ensure a secure and clutter-free email experience. With an intuitive interface built using **Streamlit**, you can classify emails with ease and view model performance metrics for transparency.

---

## 🚀 Features

- **Email Classification**: Easily classify emails as either **Spam** or **Not Spam** by entering their content.
- **Performance Insights**:
  - View model accuracy.
  - Analyze the confusion matrix.
  - Explore a detailed classification report.
- **Interactive Interface**:
  - User-friendly and intuitive design.
  - Results are displayed with a visually distinct background:
    - **Green** for Not Spam.
    - **Red** for Spam.

---

## 📦 Installation

### Step 1: Clone the Repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/Parthivkoli/SpamSearchAi.git
cd SpamSearchAi
```

### Step 2: Install Dependencies

Ensure you have **Python 3.7+** installed. Use the following command to install the required packages:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

## 🛠 Usage

1. Enter the content of the email you want to classify.
2. Click the **Classify** button.
3. View the result along with the confidence score:
   - **Spam**: Highlighted in red.
   - **Not Spam**: Highlighted in green.

---

## 📊 Model Performance

- **Accuracy**: Achieves high accuracy on the test dataset.
- **Confusion Matrix**: Provides a breakdown of true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Offers insights into precision, recall, and F1-score for both Spam and Not Spam classes.

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 💬 Support

For questions or support, please reach out to [Parthiv Koli](mailto:parthivk2310@gmail.com).

---

## 🌟 Acknowledgments

- **Streamlit**: For providing a seamless interface for web applications.
- **Scikit-learn**: For robust machine learning algorithms.
- **Python**: For being the backbone of this project.
