import streamlit as st
import pickle

# Load the model and vectorizer
try:
    model = pickle.load(open('spam.pkl', 'rb'))
    cv = pickle.load(open('vec.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please upload 'spam.pkl' and 'vec.pkl'.")

# Main function for the Streamlit app
def main():
    # Set up the page title and description
    st.title("📧 Email Spam Classification Application")
    st.markdown("""
        This **Machine Learning** application classifies emails as:
        - **Spam**: Unwanted or junk email
        - **Ham**: Legitimate or useful email
    """)
    
    # Input text area for email content
    st.subheader("Email Classification")
    user_input = st.text_area("Enter the email content to classify:", height=150)
    
    # Process and classify the input on button click
    if st.button("Classify"):
        if user_input.strip():
            try:
                # Prepare input for model
                data = [user_input]
                vec = cv.transform(data).toarray()
                
                # Predict the class
                result = model.predict(vec)
                confidence = model.predict_proba(vec).max() * 100
                
                # Display results
                st.write(f"### Input Email:\n\n{user_input}")
                if result[0] == 0:
                    st.success(f"✅ This is Not a Spam Email ({confidence:.2f}% confidence).")
                else:
                    st.error(f"🚫 This is a Spam Email ({confidence:.2f}% confidence).")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter email content to classify.")
    
    # Footer section
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit")

# Run the main function
if __name__ == "__main__":
    main()
