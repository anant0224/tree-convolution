sentences = open('/u/a/n/anant/Downloads/conll14st-test-data/noalt/official-2014.combined.m2').read().split('\n')
test_data = open('test_data', 'w')

for sentence in sentences:
	if sentence != "" and sentence[0] == 'S':
		test_data.write(sentence[2:] + '\n')

test_data.close()

