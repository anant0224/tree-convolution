preps = ['at', 'on', 'in', 'by', 'for', 'against', 'to', 'from', 'between', 'during', 'with', 'about', 'of']

annotations = open('/u/a/n/anant/Downloads/conll14st-test-data/noalt/official-2014.combined.m2').read().split('\n')
new_annt = []
for annotation in annotations:
	new_annotation = annotation.split(' ')
	new_annt.append(new_annotation)
annotations = new_annt
sentences = open('test_data_dir/words').read().decode('UTF-8').split('\n')
sentences.pop()
tags = open('test_data_dir/tags').read().split('\n')
tags.pop()
labels = open('test_data_dir/labels').read().split('\n')
labels.pop()
adj = open('test_data_dir/adj').read().split('\n')
adj.pop()
new_words = open('generated_test_data/words', 'w')
new_tags = open('generated_test_data/tags', 'w')
new_labels = open('generated_test_data/labels', 'w')
new_adj = open('generated_test_data/adj', 'w')
new_indices = open('generated_test_data/indices', 'w')
new_ylabels = open('generated_test_data/ylabels', 'w')
new_ypreps = open('generated_test_data/ypreps', 'w')

i = 0
j = 0
l = 0
h = 0
corrections = 0
for sentence in sentences:
	print "sentence: " + sentence
	while annotations[h] == '' or annotations[h][0] != 'S':
		h += 1
	# print annotations[h]
	# print "sentence: " + sentence
	h+= 1

	
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
			while annotations[h][0] != "" and annotations[h][0] == 'A' and int(annotations[h][1]) < m:
				# print annotations
				h += 1
			true_index = index
			if annotations[h][0] != "" and annotations[h][0] == 'A' and int(annotations[h][1]) == m:
				splits = annotations[h][2].split("|||")
				if int(splits[0]) == m+1:
					if splits[1] == "Prep" and splits[2] != "":
						print "annotations: " + str(annotations[h])
						bla = splits[2].lower()
						if bla in preps:
							corrections += 1
							true_index = preps.index(bla)
							# true_index = -1
						else:
							true_index = -1
				h += 1
			while annotations[h] != "" and annotations[h][0] == 'A' and int(annotations[h][1]) <= m+1:
				if int(annotations[h][1]) == m+1:
					true_index = -1
				h += 1
			if true_index != -1:
				write_words = sentence.split(' ')
				write_words[m] = '_'
				write_sentence = " ".join(write_words)
				new_words.write(write_sentence.encode('UTF-8') + '\n')
				new_tags.write(tags[i] + '\n')
				new_labels.write(labels[i] + '\n')
				new_adj.write(adj[i] + '\n')
				new_indices.write(str(m) + '\n')
				new_ylabels.write(str(true_index) + '\n')
				new_ypreps.write(preps[true_index] + '\n')
			# print word
		m += 1
	# if k == 0:
	# 	print sentence
	l += k
	i += 1

# print "corrections: " + str(corrections)

new_words.close()
new_tags.close()
new_labels.close()
new_adj.close()
new_indices.close()
new_ylabels.close()
new_ypreps.close()

