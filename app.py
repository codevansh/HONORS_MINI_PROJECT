import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Diabetes Prediction System",
    layout="wide"
)


model = joblib.load(
    "model/diabetes_logistic_model.pkl"
)


train_df = pd.read_csv(
    "datasets/train.csv"
)


feature_columns = [
    column
    for column in train_df.columns
    if column not in ["target", "row_idx"]
]


st.title("Diabetes Prediction System")
st.write(
    "Enter the required information below to generate "
    "a prediction using the trained Machine Learning model."
)

st.divider()


# CREATE INPUT FORM
with st.form("prediction_form"):
    st.subheader("Patient Information")

    input_data = {}

    col1, col2 = st.columns(2)

    for index, column in enumerate(feature_columns):
        
        current_column = col1 if index % 2 == 0 else col2
        
        with current_column:
            # NUMERICAL COLUMNS

            if pd.api.types.is_numeric_dtype(train_df[column]):
                median_value = train_df[column].median()
                input_data[column] = st.number_input(
                    column,
                    value=int(median_value),step=1
                )
                
            # CATEGORICAL / TEXT COLUMNS
            else:
                options = (
                    train_df[column].dropna().astype(str).unique().tolist()
                )

                options = sorted(options)
                if len(options) > 0:
                    input_data[column] = st.selectbox(column,options)

                else:
                    input_data[column] = st.text_input(column)

    st.divider()
    submitted = st.form_submit_button(
        "Predict Diabetes"
    )


if submitted:

    input_df = pd.DataFrame(
        [input_data]
    )

    try:

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        class_0_probability = probability[0]
        class_1_probability = probability[1]


        st.divider()
        st.subheader("Prediction Result")


        if prediction == 1:

            st.error("Prediction: Diabetes")

            st.write(
                f"Probability of Diabetes: "f"{class_1_probability:.2%}")

        else:

            st.success("Prediction: No Diabetes")

            st.write(
                f"Probability of No Diabetes: "f"{class_0_probability:.2%}")


        st.subheader("Prediction Probabilities")
        col1, col2 = st.columns(2)


        with col1:
            st.metric("No Diabetes Probability",f"{class_0_probability:.2%}")

        with col2:

            st.metric("Diabetes Probability",f"{class_1_probability:.2%}")


        st.write("Probability of Diabetes")
        st.progress(float(class_1_probability))

    except Exception as e:
        st.error(
            f"Prediction error: {e}"
        )