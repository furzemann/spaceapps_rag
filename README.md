

How to run:-
1. Install dependencies and use ```pip install -r requirements.txt```
Skip to 5 if you dont want to make your own database locally
2. Add the csv to your database
3. Make a pg database and change the db url in these files to the connection link
4. Run the pg.py file ```python pg.py```
5. replace "YOUR_DATABASE_URL" with connection url - postgresql://neondb_owner:npg_cZJvwbxs23YS@ep-flat-term-a1bhljd0-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
6. Run the streamlit app with ```streamlit run streamlit_app.py```


