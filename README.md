# EcoSmartX Backend API

## Project Description

## ENGLISH :
REST API developed with **FastAPI** to predict:

* the category of recyclable waste
* the estimated resale price

## FRANCAIS :
API REST développée avec **FastAPI** pour prédire :

* la catégorie des déchets recyclables
* le prix de revente estimé

---

## Architecture

```
app/
├── schemas/
├── services/
├── models/
├── data/
├── middleware/
├── main.py
```

---

## Technologies

* FastAPI
* Scikit-learn
* Pandas
* NumPy
* Joblib

---

## Launch the project

```bash
cd backend
uvicorn app.main:app --reload
```

---

## ML Models

* RandomForestClassifier
* RandomForestRegressor

---

## Team (Contributors)

* Jelliti Ismail 
* Hammami Ahmed