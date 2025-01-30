from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import streamlit as st
import pandas as pd
import torch
import inference


if __name__ == "__main__":
    # set up the page to be wide
    st.set_page_config(layout="wide")

    # sidebar (left side menu)
    with st.sidebar:
        st.title("Configuración")
        option = st.selectbox(
            "Selecciona el modelo para realizar las predicciones:",
            (
                "LATEiimas/xlm-roberta-base-robson-criteria-classification-ner-es",
                "LATEiimas/roberta-base-robson-criteria-classification-ner-es",
                "LATEiimas/bert-base-multilingual-cased-robson-criteria-classification-ner-es",
            ),
        )
        st.write(f"Modelo seleccionado:<br><b>{option}</b>", unsafe_allow_html=True)

        st.button("Ejecutar predicción", type="primary")
        st.subheader("Clasificación de Robson", divider=True)

        st.success('GRUPO 10', icon=":material/pregnancy:")

    # main content (right side)
    with st.container():
        st.title("Clasificación de Criterios de Robson")
        st.write("Este es el área principal de la aplicación.")

        # Ejemplo de columnas para organizar mejor el contenido
        col1, col2 = st.columns([2, 3])  # Ajusta la proporción entre columnas

        with col1:
            st.subheader("Nota clínica de la paciente", divider=True)
            txt = st.text_area(
                "Escribe la nota clínica",
                "Resumen de Interrogatorio , Exploración Física y / o Estado Mental : NOTA DE VALORACION URGENCIAS 06.09.22 21.00 HRS PACIENTE DE 26 AÑOS DE EDAD . APP . INTERROGADOS Y NEGADOS . HEMOTIPO ORH+ AGO . MENARCA A 13 AÑOS CICLOS 30X5 IVSA . 18 PS1 GESTA 1 . FUM . 28.11.22 EG 40.2 SDG ACUDE SIN USG . REFIERE QUE LLEVA SU CONTROL PRENATAL EN <Hospital> <Hospital> <Hospital> <Hospital> <Hospital> <Hospital> . REFIERE HIPERTENSION GESTACIONAL ACTUALMENTE EN TRATAMIENTO CON NIFEDIPINO 30 MG CADA 214 HORAS , ALFAMETILDOPA 500 MG CADA 8 HORAS A PARTIR D ELAS 25 SDG PADEXCIMEINTO ACTUAL . ACUCE POR REFERIR ACTIVIDAD UTERINA IRREGULAR QUE INICIO EL DIA DE HOY A LAS 17.00 HRS , ADECAUDOS MOVIMIENTOS FETALES , NIEGA PERDIDAS TARNSVAGINALES , NIEGA DATOS DE VASOESPASMO Y DE BAJO GASTO . EXPLORACION FISICA . CONSCIENTE ALERTA ORIENTADA ADECUADA COLORACION DE TEGUMENTOS Y MUCOSAS BIEN HIDRTADA CARDIOPULMONAR SIN COMPROMISO ABDOMEN GLOBOSO A EXPENSAS DE PANICLUO ADIPOSO Y UTERO GESTANTE FONDO UTERINO DE 38 CM , PUVI CEFALICO DORSO A LA ZIQUIRDA FCF 150 LPM , AL TACTO VAGINAL CAVIDAD EUTERMICA CERVIX POSTERIOR 3 CM DE DILATCAION 50% DE BORRAMIENTO , AMNIOS INTEGRO , VALSALVA Y ATARNIER NEGATIVOS , PRODUCTO LIBRE , PELVIS GINECOIDE EXTREMIDADES INTEGRAS LLENADO CAPILAR INMEDIATO , NO EDEMAS ROTS SIN ALTERACIONES SE VUELVE A TOMAR TA EN 125/80 . Diagnóstico : TRABAJO DE PARTO EN FASE LATENTE EMBARAZO DE 40.2 SEMANAS DE GESTACION POR FECHA DE ULTIMA MENSTRUACION GESTA 1 HIPERTENSION GESTACIONAL EN TARTAMIENTO",
                placeholder="Escriba aquí la nota clínica de la paciente.",
            )

        with col2:
            st.subheader("Predicciones", divider=True)
            st.markdown(
                """
                        Resumen de Interrogatorio , Exploración Física y / o Estado Mental : NOTA DE VALORACION URGENCIAS 06.09.22 21.00 HRS PACIENTE DE `26 AÑOS DE EDAD` . APP . INTERROGADOS Y NEGADOS . HEMOTIPO ORH+ AGO . MENARCA A 13 AÑOS CICLOS 30X5 IVSA . 18 PS1 GESTA 1 . FUM . 28.11.22 EG `40.2 SDG` ACUDE SIN USG . REFIERE QUE LLEVA SU CONTROL PRENATAL EN `<Hospital> <Hospital> <Hospital> <Hospital> <Hospital> <Hospital>` . REFIERE `HIPERTENSION GESTACIONAL` ACTUALMENTE EN TRATAMIENTO CON NIFEDIPINO `30 MG` CADA 214 HORAS , ALFAMETILDOPA `500 MG` `CADA 8 HORAS` A PARTIR D ELAS `25 SDG` PADEXCIMEINTO ACTUAL . ACUCE POR REFERIR ACTIVIDAD UTERINA IRREGULAR QUE INICIO EL DIA DE HOY A LAS 17.00 HRS , ADECAUDOS MOVIMIENTOS FETALES , NIEGA PERDIDAS TARNSVAGINALES , NIEGA DATOS DE VASOESPASMO Y DE BAJO GASTO . EXPLORACION FISICA . CONSCIENTE ALERTA ORIENTADA ADECUADA COLORACION DE TEGUMENTOS Y MUCOSAS BIEN HIDRTADA CARDIOPULMONAR SIN COMPROMISO ABDOMEN GLOBOSO A EXPENSAS DE PANICLUO ADIPOSO Y UTERO GESTANTE FONDO UTERINO DE 38 CM , `PUVI` `CEFALICO` DORSO A LA ZIQUIRDA FCF 150 LPM , AL TACTO VAGINAL CAVIDAD EUTERMICA CERVIX POSTERIOR 3 CM DE DILATCAION 50% DE BORRAMIENTO , AMNIOS INTEGRO , VALSALVA Y ATARNIER NEGATIVOS , PRODUCTO LIBRE , PELVIS GINECOIDE EXTREMIDADES INTEGRAS LLENADO CAPILAR INMEDIATO , NO EDEMAS ROTS SIN ALTERACIONES SE VUELVE A TOMAR TA EN 125/80 . Diagnóstico : TRABAJO DE PARTO EN FASE LATENTE EMBARAZO DE 40.2 SEMANAS DE GESTACION POR FECHA DE ULTIMA MENSTRUACION GESTA 1 HIPERTENSION GESTACIONAL EN TARTAMIENTO
                """
            )

    # Correr la aplicación con: streamlit run app.py
