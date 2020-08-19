import nltk
import sys
import os
import numpy as np
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)

def load_files(directory):
    """
    Given a directory name, returns a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    fileMap = {}
    files = os.listdir(directory)
    os.chdir(directory)
    for textFile in files:
        with open(textFile, encoding="utf8") as openFile:
            contents = openFile.read()
            fileMap[textFile] = contents
        openFile.close()
    return fileMap

def tokenize(document):
    """
    Given a document (represented as a string), returns a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    text = document
    # split into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalnum()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, returns a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfWords = {}
    for document in documents:
        wordList = documents[document]
        for word in wordList:
            idfCount = 0
            total = 0
            for testDocument in documents:
                if word in documents.get(testDocument):
                    idfCount += 1
            idfWords[word] = np.log(len(documents)/idfCount)
    return idfWords

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), returns a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    topFiles = []
    unrankedFiles = []
    rankingFiles = {}
    # Get a list of files which contain the words in the query
    for word in query:
        for file in files:
            if word in files[file]:
                unrankedFiles.append(file)
    # Assigning tf-idf values
    for file in unrankedFiles:
        rankingFiles[file] = 0
    for word in query:
        if word in idfs.keys():
            idf = idfs[word]
        else:
            continue
        for file in unrankedFiles:
            tf = files[file].count(word)
            w = tf*idf
            rankingFiles[file] += w
    # Get a 'n' number of the top files
    count = 0
    while count < n:
        highestValue = -1
        highestFile = None
        for file in rankingFiles:
            if rankingFiles[file] > highestValue:
                highestValue = rankingFiles[file]
                highestFile = file
        topFiles.append(highestFile)
        count += 1
    return topFiles

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), returns a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    topSentences = []
    rankingSentences = {}
    sentenceNumberMatch = {}
    # Assign each sentence a default score of zero
    for sentence in sentences:
        rankingSentences[sentence] = 0
    # Calculate the weight of each sentence based on the idf value of the word in the query
    for word in query:
        for sentence in sentences:
            if word in sentences[sentence]:
                rankingSentences[sentence] += idfs[word]
    for sentence in sentences:
        sentenceNumberMatch[sentence] = len(set(sentence).intersection(query)) / len(sentence)
    
    # Get a 'n' number of the top sentences
    count = 0
    while count < n:
        highestValue = -1
        highestSentence = None
        highestDensity = -1
        for sentence in rankingSentences:
            if rankingSentences[sentence] > highestValue:
                highestValue = rankingSentences[sentence]
                highestSentence = sentence
            elif rankingSentences[sentence] == highestValue:
                if sentenceNumberMatch[sentence] > highestDensity:
                    highestDensity = sentenceNumberMatch[sentence]
                    highestSentence = sentence
        topSentences.append(highestSentence)
        count += 1
    return topSentences

if __name__ == "__main__":
    main()
