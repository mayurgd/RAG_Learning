# STEPS TO RUN:

> change directory to python_doc_rag<br>
 create .env file similar to .env.sample in python_doc_rag folder<br>
 add GOOGLE_API_KEY key in .env file<br>


```python
>> uvicorn v5.backend.main:app --reload
>> python -m streamlit run v5/frontend/main.py
```
