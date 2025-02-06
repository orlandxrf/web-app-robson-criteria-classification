import streamlit as st
import pandas as pd
import inference


if __name__ == "__main__":
    # set up the page to be wide
    inf = inference.Inference()
    clinical_note = ""
    predictions = []
    st.set_page_config(layout="wide")

    if "clinical_note" not in st.session_state:
        st.session_state.clinical_note = ""

    if "predictions" not in st.session_state:
        st.session_state.predictions = ""

    # sidebar (left side menu)
    with st.sidebar:
        st.title("Configuración")
        model_name = st.selectbox(
            "Selecciona el modelo para realizar las predicciones:",
            (
                "LATEiimas/xlm-roberta-base-robson-criteria-classification-ner-es",
                "LATEiimas/roberta-base-robson-criteria-classification-ner-es",
                "LATEiimas/bert-base-multilingual-cased-robson-criteria-classification-ner-es",
            ),
        )
        inf.load_model(model_name=model_name)
        st.write(
            f"El modelo <br><b>{model_name}</b> ha sido cargado!",
            unsafe_allow_html=True,
        )

        if st.button("Ejecutar predicción", type="primary"):
            predictions, entities = inf.predict(text=st.session_state.clinical_note)
            st.session_state.entities = entities
            st.session_state.predictions = predictions
        st.subheader("Clasificación de Robson", divider=True)

        st.success("GRUPO 10", icon=":material/pregnancy:")

    # main content (right side)
    with st.container():
        st.title("Clasificación de Criterios de Robson")
        st.write("Este es el área principal de la aplicación.")

        # Ejemplo de columnas para organizar mejor el contenido
        col1, col2 = st.columns([2, 3])  # Ajusta la proporción entre columnas

        with col1:
            st.subheader("Nota clínica de la paciente", divider=True)
            clinical_note = st.text_area(
                "Escribe la nota clínica",
                value="",
                height=300,
                placeholder="Escriba aquí la nota clínica de la paciente.",
            )
            st.session_state.clinical_note = clinical_note

        with col2:
            st.subheader("Predicciones", divider=True)
            if "predictions" not in st.session_state:
                st.markdown("")
            else:
                st.markdown(st.session_state.predictions)

        if "entities" not in st.session_state:
            st.session_state.entities = ""
        else:
            if st.session_state.entities != "":
                df = pd.DataFrame(st.session_state.entities, columns=["Entity", "Text"])
                st.dataframe(df.style.hide(axis="index"))
    # Correr la aplicación con: streamlit run app.py
