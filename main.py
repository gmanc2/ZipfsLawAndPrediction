import heapq  # import heapq for heap arrays
import matplotlib.pyplot as plt  # import matplotlib.pyplot for graphs
from collections import Counter  # import counter for counting bags
from nltk.tokenize import sent_tokenize, word_tokenize  # import nltk for NLP tokenization
from nltk.corpus import stopwords  # import for detection of stopwords
import nltk
import numpy as np  # obligatory numpy(love it gotta use it when possible)
from adjustText import adjust_text  # adjust plot text for readability
# import requests  # not used
# from bs4 import BeautifulSoup not used, but it's for scraping urls, I initially used urls
# instead of having to resale the file everytime and I might use this later (NLP Summarization is interesting)

# download NLTK data
nltk.download('punkt')  # punkt tokenizer download
nltk.download('stopwords')  # remove stopwords download

"""
def scrape_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(paragraph.text for paragraph in paragraphs)
        return text

    except Exception as e:
        print(f"Error while scraping text from URL {url}: {e}")
        return None
"""



def read_text_file(file_name):  # read text file
    try:  # try to open with read permissions
        with open(file_name, 'r') as file:
            text = file.read()
    except FileNotFoundError:  # catch file not found errors
        print(f"Error: {file_name} not found.")
        return None
    except Exception as e:  # catch other exceptions
        print(f"Error while reading the file {file_name}: {e}")
        return None
    return text


def write_text_file(file_name, text):  # write text file
    try:  # try open with write permissions
        with open(file_name, 'w') as file:
            file.write(text)
    except Exception as e:  # catch other exceptions
        print(f"Error while writing the file {file_name}: {e}")


def calculate_word_frequencies(text):  # calculate frequency of words
    try:
        words = word_tokenize(text.lower())  # tokenize the text into words with ntlk
        # filter out the stopwords and non-alphanumeric words i.e. stopwords downloaded earlier
        words = [word for word in words if word not in stopwords.words('english') and word.isalnum()]
        # calculate the word frequencies using a counter
        word_freq = Counter(words)
    except Exception as e:  # catch other exceptions
        print(f"Error while calculating word frequencies: {e}")
        return None
    return word_freq



def plot_word_frequencies(word_freq, top_n=30):  # plot word frequencies and verify Zipf's law
    # sort the word frequencies in descending order and take the top number, so it doesn't take 5 years to complete
    sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    # calculate the ranks and frequencies for the plot return an iterable of the two objects
    ranks, frequencies = zip(*[(i + 1, f) for i, (_, f) in enumerate(sorted_freq)])
    # create the log-log plot in accordance with Zipf's law
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.loglog(ranks, frequencies, marker='o', linestyle='-', linewidth=1, markersize=6, markerfacecolor='red',
              markeredgecolor='black', color='blue')
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title("Word Frequencies vs Rank (Zipf's Law)")
    # calculate the axis limits
    xlim_min, xlim_max = int(np.floor(np.log10(min(ranks)))), int(np.ceil(np.log10(max(ranks))))
    ylim_min, ylim_max = int(np.floor(np.log10(min(frequencies)))), int(np.ceil(np.log10(max(frequencies))))
    # set the x-axis ticks and labels
    xticks = [10 ** i for i in range(xlim_min, xlim_max + 1)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    # set the y-axis ticks and labels
    yticks = [10 ** i for i in range(ylim_min, ylim_max + 1)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.grid(which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.6)  # add gridlines to the plot
    # annotate each point with its corresponding word/store labels in list
    texts = []
    for (word, _), rank, freq in zip(sorted_freq, ranks, frequencies):
        texts.append(ax.text(rank, freq, word, ha='center', va='bottom', fontsize=8))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', linewidth=1, color='red', alpha=0.3),
                lim=200)  # adjust the labels to prevent overlapping and add arrows (Took me forever)
    plt.show()  # show the plot


def summarize_text(text, word_freq, max_sentences=5):  # generate a summary of a given text based on word frequencies
    try:
        sentences = sent_tokenize(text)  # tokenize the text into sentences using ntlk
        most_common = heapq.nlargest(100, word_freq,
                                     key=word_freq.get)  # get the 100 most common words more words take longer
        sentence_scores = {}  # init array
        for sentence in sentences:
            # tokenize the sentence into words and convert to lowercase
            for word in word_tokenize(sentence.lower()):
                if word in most_common:  # check if the sentence is not in the sentence_scores dictionary
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = 0  # init the sentence score with 0 if not present in the dictionary
                    sentence_scores[sentence] += word_freq[word]  # add frequency of the word to the sentence score
        summary_sentences = heapq.nlargest(max_sentences, sentence_scores,
                                           key=sentence_scores.get)  # get the top sentences based on their scores
        summary = ' '.join(summary_sentences)  # combine the summary sentences into a summary text string
    except Exception as e:  # catch other exceptions
        print(f"Error while summarizing text: {e}")
        return None

    return summary


def main():
    # url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5172588/#:~:text=Zipf's%20law%20is%20a%20relation,Frequency%20%E2%88%9D%201%20Rank%20.' # not used
    input_file = 'input.txt'  # define the input file
    output_file = 'summary.txt'  # define the summary output file

    text = read_text_file(input_file)  # read text using the previous function
    if text is not None:
        # calculate the word frequency using the previous function
        word_freq = calculate_word_frequencies(text)
        if word_freq is not None:
            # generate the summary using the previous function
            summary = summarize_text(text, word_freq)
            # write the summary to an output file using the previous function
            write_text_file(output_file, summary)
            print(f"Summary written to {output_file}")
            # plot using the previous function
            plot_word_frequencies(word_freq, top_n=200)


if __name__ == "__main__":
    main()
