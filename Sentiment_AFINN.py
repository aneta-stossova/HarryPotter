import pandas as pd
from afinn import Afinn

dialogues = pd.read_csv("CSV_raw_data/Tables/Dialogue.csv", encoding="latin1")
fact_dialogue = pd.read_csv("CSV_draft/fact_dialogue_afinn_draft.csv", encoding="latin1")

# Afinn sentiment analyzer
afinn = Afinn()

dialogues["SENTIMENT_AFINN"] = dialogues["Dialogue"].apply(afinn.score)

sentiment_map = dict(zip(dialogues["Dialogue ID"], dialogues["SENTIMENT_AFINN"]))

fact_dialogue["SENTIMENT_AFINN"] = fact_dialogue["ID_DIALOGUE"].map(sentiment_map)

fact_dialogue.to_csv("CSV_final/fact_table_afinn_final.csv", index=False)
print("The table was successfully created")