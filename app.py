from transformers import AutoModelForTokenClassification
from robson_classification import RobsonClassification
from transformers import AutoTokenizer
import streamlit as st
import pandas as pd
import torch
import spacy
import time
import math


def split_sentence(sentence: list[str], max_len=510) -> list[list[str]]:
    """
    Get a list of sentences with up to 510 (max_len) tokens.

    Args:
        sentence (list): A sentence greater than 510 tokens

    Returns:
        sentence_list (list): A list with sentences with up to 510 tokens as max length.
    """
    sentence_list = []
    start = 0
    end = max_len
    snts_count = math.ceil(len(sentence) / max_len)
    for _ in range(snts_count):
        sentence_list.append(" ".join(sentence[start:end]))
        start = end
        end + max_len
    return sentence_list


def tokenize_text(text: str, max_len=510) -> list[list[str]]:
    """
    Split the text into sentences and get their tokens for each sentence.
    If tokens are longer than the max length, then split the sentences as required according to the max length.

    Args:
        text (str): Text that will be tokenized.
        max_len (int): Maximum length tokens allowed, default 510 tokens.

    Returns:
        sentences (list): A list with sentences with up to 510 tokens as max length.
    """
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(text, disable=["ner"])
    sentences = []
    for snt in doc.sents:
        snt_txt = [token.text for token in snt]
        if len(snt_txt) > max_len:
            for s in split_sentence(snt_txt, max_len=max_len):
                sentences.append(s)
        else:
            snt_txt = " ".join([token.text for token in snt])
            sentences.append(snt_txt)
    return sentences


@st.cache_resource
def load_model(model_name: str, device: str) -> None:
    """
    Load the model and tokenizer

    Args:
        model_name (str): The model name to load
        device (str): Device available to use in the inference

    Returns:
        None
    """
    st.cache_resource.clear()
    with st.spinner(
        "Descargando el modelo...",
    ):
        # downloading the model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    return model, tokenizer


def get_entities(word_labels: tuple[str, str]) -> list[tuple[str, str]]:
    """
    Retrieve the entities predicted by the model

    Args:
        word_labels (tuple): A list with the word and correspondig label [(word, label)]

    Returns:
        result (tuple): A list with the idnetified entities: [("label", "entity text")]
    """
    entities = []
    entity = {}
    result = {}
    for word, label in word_labels:
        if "O" not in label:
            start, tag = label.split("-")
            if start == "B":
                if len(entity) > 0:
                    entities.append(entity)
                    entity = {}
                entity[tag] = word
            elif start == "I":
                if (
                    tag not in entity
                ):  # a new entity without the B-tag, all tags with I-tag only
                    entities.append(entity)
                    entity = {}
                    entity[tag] = word
                else:  # entity with a previous B-tag and here with I-tags
                    entity[tag] += f" {word}"
        else:
            if len(entity) > 0:
                entities.append(entity)
                entity = {}

    for e in entities:
        if len(e) > 0:
            tag = list(e.keys())[0]
            txt = list(e.values())[0]
            if tag not in result:
                result[tag] = [txt]
            else:
                result[tag].append(txt)
            # result.append((list(e.keys())[0], list(e.values())[0]))
    return result


def align_bert_entities(aligned_labels: tuple[str, str]) -> dict[str, list[str]]:
    merged_entities = {}
    current_tokens = []
    current_label = None

    for token, label in aligned_labels:
        if token.startswith("##"):  # it is a subtoken
            current_tokens[-1] += token[2:]  # add to last token without "##"
        else:
            if label.startswith("B-") or label == "O":
                # if there is a previous entity, save it before starting a new one
                if current_tokens and current_label:
                    # merged_entities.append((" ".join(current_tokens), current_label))
                    if current_label not in merged_entities:
                        merged_entities[current_label] = [" ".join(current_tokens)]
                    else:
                        merged_entities[current_label].append(" ".join(current_tokens))
                # start new entity
                current_tokens = [token]
                current_label = (
                    label.replace("B-", "") if label.startswith("B-") else None
                )
            elif label.startswith("I-") and current_label:
                current_tokens.append(token)

    # add the last entity if it exists
    if current_tokens and current_label:
        # merged_entities.append((" ".join(current_tokens), current_label))
        if current_label not in merged_entities:
            merged_entities[current_label] = [" ".join(current_tokens)]
        else:
            merged_entities[current_label].append(" ".join(current_tokens))

    return merged_entities


def xlm_roberta_predictions(model_name: str, text: str) -> tuple[str, dict[str, str]]:
    """
    Make predictions on each text sentence.

    Args:
        text (str): Text used to make predictions

    Returns:
        final_sentences (str): Contains the text and entities according to markdown style to show code snippets.
        entity_dict (dict): A dictionary, where keys are entity tags and values contains a texts list.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_name, device)
    sentences = tokenize_text(text=text)
    final_sentences = []
    entity_dict = {}
    for sentence in sentences:
        inputs = tokenizer(
            sentence, return_tensors="pt", truncation=True, padding=True
        ).to(device)

        # get model predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # get logits and convert them to predicted token class ids
        logits = outputs.logits
        predicted_token_class_ids = torch.argmax(logits, dim=-1).squeeze().tolist()

        # get corresponding labels between ids and labels
        id_to_label = model.config.id2label

        # decoding the tokens and removing the underscore "▁" used by XLM-RoBERTa
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

        # join subtokens to form words and get the predicted labels
        word_labels = []
        current_word = ""
        current_label = None

        for token, label_id in zip(tokens, predicted_token_class_ids):
            if token == "<s>" or token == "</s>":
                continue

            label = id_to_label[label_id]

            # if token starts with "▁", it is a new token
            if token.startswith("▁"):
                if current_word:  # save the previous word before starting a new one
                    word_labels.append((current_word, current_label))

                current_word = token.replace("▁", "")  # remove "▁" from the token
                current_label = label  # assign the label of the first subtoken
            else:
                current_word += token  # concatenate subtokens without "▁"

        # save the last word
        if current_word:
            word_labels.append((current_word, current_label))

        # show the predicted labels
        result = []  # store the sentence text
        for word, _ in word_labels:
            result.append(word)
        result = " ".join(result)

        # retrieve entitys recogized and store
        entities = get_entities(word_labels)
        for k, v in entities.items():
            if k in entity_dict:
                entity_dict[k] += v
            else:
                entity_dict[k] = v

        # transform the entity text to markdown sintaxix adding symbols (``)
        for _, sub_list in entities.items():
            for ent_text in sub_list:
                result = result.replace(ent_text, f"``{ent_text}``")

        final_sentences.append(result)  # store all sentences text

    final_sentences = " ".join(final_sentences)

    return final_sentences, entity_dict


def roberta_predictions(model_name: str, text: str) -> tuple[str, dict[str, str]]:
    """
    Make predictions on each text sentence.

    Args:
        text (str): Text used to make predictions

    Returns:
        final_sentences (str): Contains the text and entities according to markdown style to show code snippets.
        entity_dict (dict): A dictionary, where keys are entity tags and values contains a texts list.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_name, device)
    sentences = tokenize_text(text=text)
    final_sentences = []
    entity_dict = {}
    for sentence in sentences:
        encoding = tokenizer(
            sentence,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offsets = encoding["offset_mapping"][0].tolist()

        # Inferencia
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Obtener las etiquetas predecidas
        predictions = torch.argmax(logits, dim=2)[0].tolist()
        pred_labels = [model.config.id2label[idx] for idx in predictions]

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])  # Tokens generados
        word_labels = []
        current_word = ""
        current_label = None

        # Alinear tokens con palabras originales
        for token, (start, end), label in zip(tokens, offsets, pred_labels):
            if start == 0 and end == 0:
                continue  # Ignorar tokens especiales (<s>, </s>)

            clean_token = token.lstrip("Ġ")  # Eliminar prefijo "Ġ" de nuevas palabras
            decoded_token = tokenizer.convert_tokens_to_string(
                [clean_token]
            ).strip()  # Decodificar correctamente

            if start == 0 or token.startswith("Ġ"):  # Nuevo inicio de palabra
                if current_word:
                    word_labels.append(
                        (current_word, current_label)
                    )  # Guardar la palabra anterior
                current_word = decoded_token
                current_label = label
            else:
                current_word += decoded_token  # Concatenar fragmentos de palabras

        # Agregar la última palabra
        if current_word:
            word_labels.append((current_word, current_label))

        # show the predicted labels
        result = []  # store the sentence text
        for word, _ in word_labels:
            result.append(word)
        result = " ".join(result)

        # retrieve entitys recogized and store
        entities = get_entities(word_labels)
        for k, v in entities.items():
            if k in entity_dict:
                entity_dict[k] += v
            else:
                entity_dict[k] = v

        # transform the entity text to markdown sintaxix adding symbols (``)
        for _, sub_list in entities.items():
            for ent_text in sub_list:
                result = result.replace(ent_text, f"``{ent_text}``")

        final_sentences.append(result)  # store all sentences text

    final_sentences = " ".join(final_sentences)
    return final_sentences, entity_dict


def bert_predictions(model_name: str, text: str) -> tuple[str, dict[str, str]]:
    """
    Make predictions on each text sentence.

    Args:
        text (str): Text used to make predictions

    Returns:
        final_sentences (str): Contains the text and entities according to markdown style to show code snippets.
        entity_dict (dict): A dictionary, where keys are entity tags and values contains a texts list.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_name, device)
    sentences = tokenize_text(text=text)
    final_sentences = []
    entity_dict = {}

    for snt in sentences:
        inputs = tokenizer(snt, return_tensors="pt", truncation=True, padding=True).to(
            device
        )

        # get model predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # get logits and convert them to predicted token class ids
        logits = outputs.logits

        # get the predicted labels (argmax over the logits)
        predictions = torch.argmax(logits, dim=2)

        # convert tag IDs to tag names
        predicted_labels = [
            model.config.id2label[label_id] for label_id in predictions[0].tolist()
        ]

        # align labels with original tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        aligned_labels = []
        for token, label in zip(tokens, predicted_labels):
            # ignore special tokens like [CLS], [SEP], etc.
            if token not in tokenizer.all_special_tokens:
                aligned_labels.append((token, label))

        # retrieve entitys recogized and store
        entities = align_bert_entities(aligned_labels)
        for k, v in entities.items():
            if k in entity_dict:
                entity_dict[k] += v
            else:
                entity_dict[k] = v

        # retrieve the original text aligned
        original_text = []
        for token, label in aligned_labels:
            if token.startswith("##"):  # it is a subtoken
                original_text[-1] += token[2:]  # concatenate without "##"
            else:
                original_text.append(token)
        original_text = " ".join(original_text)

        # transform the entity text to markdown sintaxix adding symbols (``)
        for _, sub_list in entities.items():
            for ent_text in sub_list:
                original_text = original_text.replace(ent_text, f"``{ent_text}``")

        final_sentences.append(original_text)  # store all sentences text

    final_sentences = " ".join(final_sentences)

    return final_sentences, entity_dict
    # corregir la función que obtiene el grupo de robson


def stream_clinical_note(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


if __name__ == "__main__":
    previous_name_model = None
    clinical_note = ""
    rc = RobsonClassification()
    rc.load_robson_group_descriptions(json_path="10_groups.json")
    # set up the page to be wide
    # Función para descargar y cargar el modelo
    clinical_note = ""
    predictions = []
    st.set_page_config(layout="wide")
    loading_message = st.empty()

    if "clinical_note" not in st.session_state:
        st.session_state.clinical_note = ""

    if "predictions" not in st.session_state:
        st.session_state.predictions = ""

    if "classification" not in st.session_state:
        st.session_state.classification = {}

    if "trigger_btn" not in st.session_state:
        st.session_state.trigger_btn = False

    # sidebar (left side menu)
    with st.sidebar:
        st.title("Configuración")
        model_name = st.radio(
            "Selecciona un modelo",
            [
                "LATEiimas/xlm-roberta-base-robson-criteria-classification-ner-es",
                "LATEiimas/roberta-base-robson-criteria-classification-ner-es",
                "LATEiimas/bert-base-robson-criteria-classification-ner-es",
            ],
            captions=[
                "XLM-RobERTa",
                "RoBERTa-Biomedical",
                "BERT-multilingual",
            ],
            index=None,
        )

        if st.button("Ejecutar predicción", type="primary"):
            st.session_state.predictions = ""
            st.session_state.classification = {}
            if model_name != None:
                if "LATEiimas/xlm-roberta" in model_name:
                    predictions, entities = xlm_roberta_predictions(
                        model_name=model_name, text=st.session_state.clinical_note
                    )
                    st.session_state.entities = entities
                    st.session_state.predictions = predictions
                    st.session_state.classification = rc.get_group(entities)
                    st.session_state.clinical_note = ""
                elif "LATEiimas/roberta" in model_name:
                    predictions, entities = roberta_predictions(
                        model_name=model_name, text=st.session_state.clinical_note
                    )
                    st.session_state.entities = entities
                    st.session_state.predictions = predictions
                    st.session_state.classification = rc.get_group(entities)
                    st.session_state.clinical_note = ""
                elif "LATEiimas/bert" in model_name:
                    predictions, entities = bert_predictions(
                        model_name=model_name, text=st.session_state.clinical_note
                    )
                    st.session_state.entities = entities
                    st.session_state.predictions = predictions
                    st.session_state.classification = rc.get_group(entities)
                    st.session_state.clinical_note = ""

    # main content (right side)
    with st.container():
        st.title("Clasificación de Criterios de Robson")

        # Ejemplo de columnas para organizar mejor el contenido
        col1, col2 = st.columns([2, 3])  # Ajusta la proporción entre columnas

        with col1:
            st.subheader("Nota clínica de la paciente", divider=True)
            clinical_note = st.text_area(
                "Escribe la nota clínica",
                value="",
                height=350,
                placeholder="Escriba aquí la nota clínica de la paciente.",
            )
            st.session_state.clinical_note = clinical_note

        with col2:
            st.subheader("Predicciones", divider=True)
            if "predictions" not in st.session_state:
                st.markdown("")
            else:
                st.write_stream(stream_clinical_note(st.session_state.predictions))

    with st.container():
        if "entities" not in st.session_state:
            st.session_state.entities = ""
        else:
            if st.session_state.entities != "":
                st.subheader("Entidades identificadas", divider=True)
                df = pd.DataFrame(
                    dict(
                        [
                            (key, pd.Series(value))
                            for key, value in st.session_state.entities.items()
                        ]
                    )
                )
                st.dataframe(df.style.hide(axis="index"), use_container_width=True)

        st.divider()

    with st.container():
        col_1, col_2 = st.columns([0.1, 0.9])
        with col_1:
            if len(st.session_state.classification) > 0:
                st.subheader(f"GRUPO {st.session_state.classification['group']}")
                st.image(
                    f"imgs/{st.session_state.classification['group']}.png",
                    width=150,
                    caption=f"GRUPO {st.session_state.classification['group']}",
                )

        with col_2:
            if len(st.session_state.classification) > 0:
                st.subheader(f"Descripción del grupo")
                st.markdown(st.session_state.classification["description"])

    # Correr la aplicación con: streamlit run app.py
