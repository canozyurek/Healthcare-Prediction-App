from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
from pydantic import BaseModel
from catboost import CatBoostClassifier
from helper import transformations
import joblib

app = FastAPI(title="Health Issues Prediction System")


class Data(BaseModel):
    gender: str
    age: int
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


@app.get("/")
@app.get("/home")
def read_home():

    return {"message": "System is healthy"}


@app.post("/predict-stroke")
def predict_stroke(data: Data):
    data_frame = pd.DataFrame(
        data=np.array(
            [
                data.gender,
                data.age,
                data.hypertension,
                data.heart_disease,
                data.ever_married,
                data.work_type,
                data.Residence_type,
                data.avg_glucose_level,
                data.bmi,
                data.smoking_status,
            ]
        ).reshape(1, 10),
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
        ],
    )
    data_frame.reset_index(inplace=True)
    stroke_model = CatBoostClassifier()
    stroke_model.load_model("models/stroke_model")
    data_frame_transformed = transformations.stroke_data_transform(data_frame)
    prediction = stroke_model.predict_proba(data_frame_transformed)[0][1]
    return round(float(prediction), 2)


class DataHyp(BaseModel):
    gender: str
    age: int
    heart_disease: str
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


@app.post("/predict-hypertension")
async def predict_hypertension(data: DataHyp):
    data_frame = pd.DataFrame(
        data=np.array(
            [
                data.gender,
                data.age,
                data.heart_disease,
                data.ever_married,
                data.work_type,
                data.Residence_type,
                data.avg_glucose_level,
                data.bmi,
                data.smoking_status,
            ]
        ).reshape(1, 9),
        columns=[
            "gender",
            "age",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
        ],
    )
    with open("hypertension_model", "rb") as hyp_loc:
        hyp_model = joblib.load(hyp_loc)
    data_frame_transformed = transformations.hypertension_data_transform(data_frame)
    prediction = hyp_model.predict_proba(data_frame_transformed)[0][1]
    return round(float(prediction), 2)


class DataGlu(BaseModel):
    gender: str
    age: int
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    Residence_type: str
    bmi: float
    smoking_status: str


@app.post("/predict-glucose")
async def predict_glucose(data: DataGlu):
    data_frame = pd.DataFrame(
        data=np.array(
            [
                data.gender,
                data.age,
                data.hypertension,
                data.heart_disease,
                data.ever_married,
                data.work_type,
                data.Residence_type,
                data.bmi,
                data.smoking_status,
            ]
        ).reshape(1, 9),
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "bmi",
            "smoking_status",
        ],
    )
    with open("models/glucose_model.sav", "rb") as glu_path:
        glu_model = joblib.load(glu_path)
    data_frame_transformed = transformations.glucose_data_transform(data_frame)
    prediction = glu_model.predict(data_frame_transformed)
    return round(float(prediction), 2)


class DataBmi(BaseModel):
    gender: str
    age: int
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    smoking_status: str


@app.post("/predict-bmi")
def predict_bmi(data: DataBmi):
    data_frame = pd.DataFrame(
        data=np.array(
            [
                data.gender,
                data.age,
                data.hypertension,
                data.heart_disease,
                data.ever_married,
                data.work_type,
                data.Residence_type,
                data.avg_glucose_level,
                data.smoking_status,
            ]
        ).reshape(1, 9),
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "smoking_status",
        ],
    )
    with open("models/bmi_model.sav", "rb") as bmi_path:
        bmi_model = joblib.load(bmi_path)
    data_frame_transformed = transformations.bmi_data_transform(data_frame)
    prediction = bmi_model.predict(data_frame_transformed)
    return round(float(prediction), 2)


class DataHypGlu(BaseModel):
    gender: str
    age: int
    heart_disease: str
    ever_married: str
    work_type: str
    Residence_type: str
    bmi: float
    smoking_status: str


@app.post("/predict-hypglu")
def predict_hypglu(data: DataHypGlu):
    data_frame = pd.DataFrame(
        data=np.array(
            [
                data.gender,
                data.age,
                "No",
                data.heart_disease,
                data.ever_married,
                data.work_type,
                data.Residence_type,
                1,
                data.bmi,
                data.smoking_status,
            ]
        ).reshape(1, 10),
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
        ],
    )
    hyp_model = CatBoostClassifier()
    with open("models/hyp_glu_dual/hyp_model.sav", "rb") as hyp_path:
        hyp_model = joblib.load(hyp_path)
    with open("models/hyp_glu_dual/glu_model.sav", "rb") as glu_path:
        glu_model = joblib.load(glu_path)

    data_1, data_2 = transformations.hyp_glu_transformation(data_frame)
    prediction_hyp = hyp_model.predict_proba(data_1)[0][1]
    prediction_glu = glu_model.predict(data_2)[0]
    return round(float(prediction_hyp), 2), round(float(prediction_glu), 2)


class DataHypBmi(BaseModel):
    gender: str
    age: int
    heart_disease: str
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    smoking_status: str


@app.post("/predict-hypbmi")
def predict_hypbmi(data: DataHypBmi):
    data_frame = pd.DataFrame(
        data=np.array(
            [
                data.gender,
                data.age,
                "No",
                data.heart_disease,
                data.ever_married,
                data.work_type,
                data.Residence_type,
                data.avg_glucose_level,
                1,
                data.smoking_status,
            ]
        ).reshape(1, 10),
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
        ],
    )
    with open("models/hyp_bmi_dual/hyp_model.sav", "rb") as hyp_path:
        hyp_model = joblib.load(hyp_path)
    with open("models/hyp_bmi_dual/bmi_model.sav", "rb") as bmi_path:
        bmi_model = joblib.load(bmi_path)

    data_1, data_2 = transformations.hyp_bmi_transformation(data_frame)
    prediction_hyp = hyp_model.predict_proba(data_1)[0][1]
    prediction_bmi = bmi_model.predict(data_2)
    return round(float(prediction_hyp), 2), round(float(prediction_bmi), 2)


class DataGluBmi(BaseModel):
    gender: str
    age: int
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str


@app.post("/predict-glubmi")
def predict_glubmi(data: DataGluBmi):
    data_frame = pd.DataFrame(
        data=np.array(
            [
                data.gender,
                data.age,
                data.hypertension,
                data.heart_disease,
                data.ever_married,
                data.work_type,
                data.Residence_type,
                1,
                1,
                data.smoking_status,
            ]
        ).reshape(1, 10),
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
        ],
    )

    with open("models/glu_bmi_dual/glu_model.sav", "rb") as glu_path:
        glu_model = joblib.load(glu_path)
    with open("models/glu_bmi_dual/bmi_model.sav", "rb") as bmi_path:
        bmi_model = joblib.load(bmi_path)
    data_1, data_2 = transformations.glu_bmi_transformation(data_frame)
    prediction_glu = glu_model.predict(data_1)
    prediction_bmi = bmi_model.predict(data_2)
    return round(float(prediction_glu), 2), round(float(prediction_bmi), 2)


class DataHypGluBmi(BaseModel):
    gender: str
    age: int
    heart_disease: str
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str


@app.post("/predict-hypglubmi")
def predict_hypglubmi(data: DataHypGluBmi):
    data_frame = pd.DataFrame(
        data=np.array(
            [
                data.gender,
                data.age,
                "No",
                data.heart_disease,
                data.ever_married,
                data.work_type,
                data.Residence_type,
                1,
                1,
                data.smoking_status,
            ]
        ).reshape(1, 10),
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
        ],
    )

    with open("models/glu_bmi_hyp/hyp_model.sav", "rb") as hyp_path:
        hyp_model = joblib.load(hyp_path)
    with open("models/glu_bmi_hyp/glu_model.sav", "rb") as glu_path:
        glu_model = joblib.load(glu_path)
    with open("models/glu_bmi_hyp/bmi_model.sav", "rb") as bmi_path:
        bmi_model = joblib.load(bmi_path)
    data_1, data_2, data_3 = transformations.hyp_glu_bmi_transformation(data_frame)
    prediction_hyp = hyp_model.predict_proba(data_2)[0][1]
    prediction_glu = glu_model.predict(data_1)
    prediction_bmi = bmi_model.predict(data_3)
    return (
        round(float(prediction_hyp), 2),
        round(float(prediction_glu), 2),
        round(float(prediction_bmi), 2),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
