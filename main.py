import streamlit as st


class App:
    def __init__(self):
        pass

    def website(self):
        st.title("Welcome to Intraday stock app 📊")
        st.write('''The current app extracts live stock data from alpha vantage
        api and then does the data preprocessing and performs model prediction of the closing prices
        of the stock symbol selected by the user.''')
        st.write('''Select the stock symbol and then analysis of the dta starts displaying the predicted values and the actual value
        of closing.''')
        if "text_input" not in st.session_state:
            st.session_state['text_input'] = ""




def main():
    st.set_page_config(page_title="Intraday stock app")
    app = App()
    app.website()

if __name__=="__main__":
    main()


