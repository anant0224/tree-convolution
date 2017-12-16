preps = ['at', 'on', 'in', 'by', 'for', 'against', 'to', 'from', 'between', 'during', 'with', 'about', 'of']

sentences = open('words').read().decode('UTF-8').split('\n')
tags = open('tags').read().split('\n')
labels = open('labels').read().split('\n')
adj = open('adj').read().split('\n')
new_words = open('data/words', 'w')
new_tags = open('data/tags', 'w')
new_labels = open('data/labels', 'w')
new_adj = open('data/adj', 'w')
new_indices = open('data/indices', 'w')
new_ylabels = open('data/ylabels', 'w')

i = 0
j = 0
l = 0
for sentence in sentences:
	# print i
	words = sentence.split(' ')
	k = 0
	m = 0
	for word in words:
		word = word.lower()
		if word in preps:
			if k == 0:
				k = 1
			j += 1
			index = preps.index(word)
			new_words.write(sentence.encode('UTF-8') + '\n')
			new_tags.write(tags[i] + '\n')
			new_labels.write(labels[i] + '\n')
			new_adj.write(adj[i] + '\n')
			new_indices.write(str(m) + '\n')
			new_ylabels.write(str(index) + '\n')
			# print word
		m += 1
	if k == 0:
		print sentence
	l += k
	i += 1

new_words.close()
new_tags.close()
new_labels.close()
new_adj.close()
new_indices.close()
new_ylabels.close()

