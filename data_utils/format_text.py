from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from os import listdir
import re

path = '/u/a/n/anant/Downloads/2539/2539/download/CORPUS_ByDisc/AH/English'
pattern = '.*_(.*)\.xml'
filenames = []

for string in listdir(path):
	filenames.append(re.match(pattern, string).group(1) + '.txt')

print(len(filenames))

newpath = '/u/a/n/anant/Downloads/2539/2539/download/CORPUS_TXT/'
lines = []

for string in filenames:
	doc = open(newpath+string).read().decode('utf-8')
	doc = re.sub('<quote>.*?</quote>', '', doc)
	doc = re.sub('<fnote>.*?</fnote>', '', doc)
	doc = re.sub('<heading>.*?</heading>', '', doc)
	doc = re.sub('<enote>.*?</enote>', '', doc)
	doc = re.sub('<picture/>', '', doc)
	doc = re.sub('\(.*?\)', '', doc)



	sentences = sent_tokenize(doc)
	for sentence in sentences:
		print(sentence)
	lines = lines + sentences

open('ground_truth.txt', 'w').write("\n".join(lines).encode('utf-8'))

