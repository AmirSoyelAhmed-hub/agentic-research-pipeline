import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="RL Research Agent", page_icon="🔬")

st.title("🔬 RL Research Agent")
st.caption("Ask questions about recent reinforcement learning papers, grounded in real arXiv data.")

question = st.text_input("Ask a question:", placeholder="e.g. What are recent approaches to reward shaping?")

if st.button("Ask") and question:
    with st.spinner("Retrieving papers and generating answer..."):
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=60)
            response.raise_for_status()
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Sources")
            for source in data["sources"]:
                st.markdown(f"- [{source['title']}]({source['url']})")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")