import os
import pandas as pd
# Imports the Google Cloud Translation library
from google.cloud import translate


# Initialize Translation client
def translateWord(lang,text="YOUR_TEXT_TO_TRANSLATE", project_id="crucial-raceway-334422"):
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": lang,
            "target_language_code": "en-US",
        }
    )
    for translation in response.translations:
        return translation.translated_text

    # Display the translation for each input text provided
    # for translation in response.translations:
    #  print(translation)
    #  print("Translated text: {}".format(translation.translated_text))


def translateBatch(lang='fr', 
    input_dir='crawler/keywords_generation/keywords_fasttext',
    output_dir='crawler/keywords_generation/googleTrans/'):
    file_path = os.path.join(input_dir, lang+'.csv')
    df = pd.read_csv(file_path, header=None)

    google_translated = []
    for word in df[0]:
        translated = translateWord(lang=lang, text=word)
        print(word, '--->', translated)
        google_translated.append(translated)

    df['google_translated']= google_translated
    df.to_csv(os.path.join(output_dir, lang+'.csv'),index=False)


if __name__ == '__main__':
    input_dir = 'crawler/keywords_generation/keywords_fasttext'
    for file in os.listdir(input_dir):
        if file.endswith('no.csv'):
            lang = file.replace('.csv','')
            if lang != 'en':
                print(file)
                translateBatch(lang=lang)