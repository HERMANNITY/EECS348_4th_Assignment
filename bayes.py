# Name: 
# Date:
# Description:
#
#

import math, os, pickle, re
from random import shuffle

class Bayes_Classifier:

	def __init__(self, fold = 1):
		"""This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
		cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
		the system will proceed through training.  After running this method, the classifier 
		is ready to classify input text."""
		self.process_filenames()
		if fold > 1:			# If cross-validation enabled
			self.cross_validate_train(fold)
		else:					# Train on whole data set
			if os.path.isfile("PosDic.txt") and os.path.isfile("NegDic.txt"):  # pickled files exist, load into memory
				self.PosList = self.load("PosList.txt")
				self.NegList = self.load("NegList.txt")
				self.PosDic = self.load("PosDic.txt")
				self.NegDic = self.load("NegDic.txt")
			else:              # pickled files do no exist, train the system
				self.train(fold)
				self.compute()

	def compute(self):
		"""Compute values for member variables"""
		# prior probability of class Positive
		self.pos_prior = float(len(self.PosList))/(len(self.PosList)+len(self.NegList))  
		print "P(positive) = " + str(self.pos_prior)
		# prior probability of class Negative
		self.neg_prior = 1 - self.pos_prior
		print "P(negative) = " + str(self.neg_prior)

		self.PosValues = 0.0	# Reset self.PosValues and self.NegValues	
		self.NegValues = 0.0
		for i in xrange(len(self.PosDic.values())):
			self.PosValues += self.PosDic.values()[i]    # sum of PosDic values
		print "Positive frequency = " + str(self.PosValues)
		for i in xrange(len(self.NegDic.values())):
			self.NegValues += self.NegDic.values()[i]    # sum of NegDic values
		print "Negative frequency = " + str(self.NegValues)
		print ""

	def process_filenames(self):
		"""Store all filenames into a list and partition them into pos. and neg."""
		self.PosList = []
		self.NegList = []
		IFileList = []
		for fFileObj in os.walk("movies_reviews/"):
			IFileList = fFileObj[2]
			break
		for i in xrange(len(IFileList)):
			if IFileList[i][7] == '1':
				self.NegList.append(IFileList[i])
			elif IFileList[i][7] == '5':
				self.PosList.append(IFileList[i])
		self.PosList = self.PosList[:len(self.NegList)]	# Make sure we have the same number of pos. and neg. files

	def train(self, fold):   
		"""Trains the Naive Bayes Sentiment Classifier."""
		self.PosDic = {}
		self.NegDic = {}
		# put into two dictionaries: Positive and Negative
		self.makeDictionary(self.PosList,self.PosDic)
		self.makeDictionary(self.NegList,self.NegDic)

		# print "PosList: " + str(len(self.PosList))
		# print "NegList: " + str(len(self.NegList))
		# print "PosDic: " + str(len(self.PosDic))
		# print "NegDic: " + str(len(self.NegDic))
		if fold == 1:	# Save dictionaries only if cv is off
			self.save(self.PosList,"PosList.txt")
			self.save(self.NegList,"NegList.txt")
			self.save(self.PosDic,"PosDic.txt")
			self.save(self.NegDic,"NegDic.txt")
			print "Successfully saved"

	def makeDictionary(self,sublist,dic):
		"""
		Given a list of file names, create a dictionary to store all of the words 
		occur in total files, and the frequencies at which they occur
		"""
		for i in xrange(len(sublist)):
			s = self.loadFile("movies_reviews/"+sublist[i])
			split = self.tokenize(s.lower())
			for j in xrange(len(split)):
				if dic.has_key(split[j]):
					dic[split[j]] += 1
				else:
					dic_add={split[j]:1}
					dic.update(dic_add)
		return dic

	def classify(self,sText):
		"""Given a target string sText, this function returns the most likely document
		class to which the target string belongs (i.e., positive, negative or neutral).
		"""

		wordlist = self.tokenize(sText.lower())
		#print wordlist
		#pos_prior = 0.0
		#neg_prior = 0.0
		pos_likelihood = 0.0
		neg_likelihood = 0.0
		p_pos = 0.0
		p_neg = 0.0
		# prior probability of class Positive
		#pos_prior = float(len(self.PosList))/(len(self.PosList)+len(self.NegList))  
		#print pos_prior
		# prior probability of class Negative
		#neg_prior = 1 - pos_prior
		#print neg_prior

		pos_likelihood = self.cal_Likelihood(self.PosDic, 1, wordlist)
		neg_likelihood = self.cal_Likelihood(self.NegDic, 0, wordlist)
		#print pos_likelihood
		#print neg_likelihood

		pos_likelihood += math.log(self.pos_prior)
		neg_likelihood += math.log(self.neg_prior)

		if pos_likelihood >= neg_likelihood: return "positive"
		else: return "negative"

		#distance = pos_likelihood - neg_likelihood
		#if distance > 1:
		#	return "positive"
		#elif distance < -1:
		#	return "negative"
		#else:
		#	return "neutral"

	def cal_Likelihood(self,dic,myclass,wordlist):
		total = 0.0
		likelihood = 0.0
		#for i in xrange(len(dic.values())):
		#   total += dic.values()[i]    # sum of Dic values
		#print total
		if myclass == 1:
			total = self.PosValues
		else:
			total = self.NegValues
		for j in wordlist:
			if j in dic:
				likelihood += math.log((dic[j]+1)/total)
			else:
				likelihood += math.log(1/total)
		return likelihood

	def cross_validate_train(self, fold):
		"""Do cross validation"""
		pos_file_list = segment(self.PosList, fold)	# A list of sublists of pos. filenames
		neg_file_list = segment(self.NegList, fold)	# A list of sublists of neg. filenames
		cv_result = [0 for x in xrange(6)]	# [pos_precision, neg_precision, pos_recall, neg_recall, pos_fm, neg_fm]
		strings = ['pos_precision', 'neg_precision', 'pos_recall', 'neg_recall', 'pos_fm', 'neg_fm']
		for i in xrange(fold):
			print "The " + str(i+1) + "th validation:"
			self.PosList = []		# Reset self.PosList
			self.NegList = []		# Reset self.NegList 
			test_data = []			# Test data for current validation
			for j in xrange(fold):	# Store training data of this validation into self.PosList and self.NegList
				if j != i:
					self.PosList += pos_file_list[j]
					self.NegList += neg_file_list[j]
				else:
					test_data += pos_file_list[j]
					test_data += neg_file_list[j]
			self.train(fold)		# Train the classifier using training data for current validation
			self.compute()
			result = self.cross_validate_test(test_data)	# Test the classifier using test data for current validation
			for k in xrange(6):		# Store result of each validation
				cv_result[k] += result[k]
		for k in xrange(6):			# Take average of all validations
			cv_result[k] /= fold
			print strings[k] + " =  " + str(cv_result[k])

	def cross_validate_test(self,sublist):
		true_pos = 0
		true_neg = 0
		false_pos = 0
		false_neg = 0
		pos_precision = 0.0 
		pos_recall = 0.0
		pos_f_measure = 0.0
		neg_precision = 0.0 
		neg_recall = 0.0
		neg_f_measure = 0.0
		for i in xrange(len(sublist)):
			s = self.loadFile("movies_reviews/"+sublist[i])
			predict = self.classify(s)
			g_truth_index = sublist[i][7]
			if g_truth_index == '1':
				g_truth = "negative"
			elif g_truth_index == '5':
				g_truth = "positive"

			if predict == g_truth and predict == "positive":
				true_pos += 1
			elif predict == g_truth and predict == "negative":
				true_neg += 1
			elif predict != g_truth and predict == "positive":
				false_pos += 1
			elif predict != g_truth and predict == "negative":
				false_neg += 1
		pos_precision = float(true_pos)/float(true_pos+false_pos)
		neg_precision = float(true_neg)/float(true_neg+false_neg)
		pos_recall = float(true_pos)/float(true_pos+false_neg)
		neg_recall = float(true_neg)/float(true_neg+false_pos)
		pos_f_measure = 2*pos_precision*pos_recall/(pos_precision+pos_recall)
		neg_f_measure = 2*neg_precision*neg_recall/(neg_precision+neg_recall)
		return (pos_precision, neg_precision, pos_recall, neg_recall, pos_f_measure, neg_f_measure)

	def loadFile(self, sFilename):
		"""Given a file name, return the contents of the file as a string."""

		f = open(sFilename, "r")
		sTxt = f.read()
		f.close()
		return sTxt
	
	def save(self, dObj, sFilename):
		"""Given an object and a file name, write the object to the file using pickle."""

		f = open(sFilename, "w")
		p = pickle.Pickler(f)
		p.dump(dObj)
		f.close()
	
	def load(self, sFilename):
		"""Given a file name, load and return the object stored in the file."""

		f = open(sFilename, "r")
		u = pickle.Unpickler(f)
		dObj = u.load()
		f.close()
		return dObj

	def tokenize(self, sText): 
		"""Given a string of text sText, returns a list of the individual tokens that 
		occur in that string (in order)."""

		lTokens = []
		sToken = ""
		for c in sText:
			if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
				sToken += c
			else:
				if sToken != "":
					lTokens.append(sToken)
					sToken = ""
				if c.strip() != "":
					lTokens.append(str(c.strip()))
					
		if sToken != "":
			lTokens.append(sToken)

		return lTokens

def segment(corpus, fold):
		shuffle(corpus)
		sublist = []
		for i in range(fold):
			sublist.append([])
		for j in range(len(corpus)):
			sublist[j%fold].append(corpus[j])
		return sublist

if __name__ == "__main__":
	bc = Bayes_Classifier(10)
