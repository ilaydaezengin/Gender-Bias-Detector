import streamlit as st
from multiapp import MultiApp

#st.markdown("# Main page of Gender Bias Tool")
#st.sidebar.markdown("# Main page 🎈")
#st.sidebar.success("Select a demo above.")

import app,info, data_tables, word_clouds, analysis
#PAGES = {
   # "Home": info,
 #   "Word Embedding Visualization": app,
   # "Gender Bias": info,
#}
#st.sidebar.title('Navigation')
#selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))
#page = PAGES[selection]
#page.app()

app_ = MultiApp()
# Add all your application here
#app_.add_app("Home", info.app)
app_.add_app("Data Analysis", data_tables.app)
app_.add_app("Embeddings",app.app)
#app_.add_app("Word Clouds", word_clouds.app)
app_.add_app('Bias Analysis with Word Embeddings', analysis.app)
# The main app
app_.run()
