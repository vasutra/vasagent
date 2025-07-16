"""Entry point for the Streamlit application."""

import streamlit as st
from pages import document_qa

st.set_page_config(page_title="Vasagent", page_icon=":robot_face:", layout="centered")

PAGES = {
    "Document QA": document_qa.show,
}


def main() -> None:
    """Run the multi-page Streamlit application."""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page()


if __name__ == "__main__":
    main()
