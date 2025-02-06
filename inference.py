from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import torch


class Inference:
    """
    Inference class to make predictions
    """

    def __init__(self) -> None:
        """
        Initialize the Inference class
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = None
        self.tokenizer = None
        self.model = None

    def load_model(self, model_name: str) -> None:
        """
        Load the model and tokenizer

        Args:
            model_name (str): The model name to load

        Returns:
            None
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(
            self.device
        )

    def get_entities(self, word_labels: tuple[str, str]) -> list[tuple[str, str]]:
        """
        Retrieve the entities predicted by the model

        Args:
            word_labels (tuple): A list with the word and correspondig label [(word, label)]

        Returns:
            result (tuple): A list with the idnetified entities: [("label", "entity text")]
        """
        entities = []
        entity = {}
        result = []
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
                result.append((list(e.keys())[0], list(e.values())[0]))
        return result

    def predict(
        self,
        text: str,
    ) -> list[str]:
        """
        Make predictions on the given text

        Args:
            text (str): The text to make predictions on

        Returns:
            list: The list of predicted labels
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(
            self.device
        )

        # get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # get logits and convert them to predicted token class ids
        logits = outputs.logits
        predicted_token_class_ids = torch.argmax(logits, dim=-1).squeeze().tolist()

        # get corresponding labels between ids and labels
        id_to_label = self.model.config.id2label

        # decoding the tokens and removing the underscore "▁" used by XLM-RoBERTa
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze().tolist()
        )

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
        result = []

        for word, label in word_labels:
            result.append(word)
        result = " ".join(result)

        entities = self.get_entities(word_labels)
        for _, ent_text in entities:
            result = result.replace(ent_text, f"``{ent_text}``")

        return result, entities
