def preProcessing(text):
    """ Preprocessing gets the text ready. Fetches all the words"""
    import re
    import string
    # Convert text to lowercase
    outText = text.lower()

    # Remove numbers
    outText = re.sub(r'\d+', '', outText)

    # Remove punctuation
    outText = outText.translate(str.maketrans("", "", string.punctuation))

    # Remove whitespaces
    outText = outText.strip()

    # Remove stopwords
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(outText)
    outText = [i for i in tokens if not i in stop_words]

    # Lemmatization
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    result = []
    for word in outText:
        result.append(lemmatizer.lemmatize(word))

    return result