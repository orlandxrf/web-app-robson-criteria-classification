from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import pipeline
import torch
from datasets import load_from_disk


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
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        # get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # get logits and convert them to predicted token class ids
        logits = outputs.logits
        predicted_token_class_ids = torch.argmax(logits, dim=-1).squeeze().tolist()

        # get corresponding labels between ids and labels
        id_to_label = self.model.config.id2label

        # decoding the tokens and removing the underscore "▁" used by XLM-RoBERTa
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        
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
            label = int(label.split('_')[1])
            result.append((word, self.model.config.id2label[label]))

        return result
