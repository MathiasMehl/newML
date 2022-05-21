from advanced_summarizer import summarize

# https://github.com/UsmanNiazi/DUC-2004-Dataset
# https://paperswithcode.com/sota/text-summarization-on-duc-2004-task-1


summary = summarize("conclusion.txt")
'''documents = ["duc2004/documents/D1.txt", "duc2004/documents/D2.txt", "duc2004/documents/D3.txt",
             "duc2004/documents/D5.txt", "duc2004/documents/D6.txt",
             "duc2004/documents/D7.txt", "duc2004/documents/D8.txt", "duc2004/documents/D9.txt",
             "duc2004/documents/D10.txt"]
for document in documents:
    summarize(document)'''
