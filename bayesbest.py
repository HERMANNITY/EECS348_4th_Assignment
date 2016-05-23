# Name: 
# Date:
# Description:
#
#

import math, os, pickle, re, nltk
from random import shuffle

class Bayes_Classifier:

   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""
      self.PosList = []
      self.NegList = []

      self.PosDic = {}             # a dictionary created by stem and unigram strategy
      self.NegDic = {}
      self.PosBigramDic = {}       # a dictionary created by bigram strategy
      self.NegBigramDic = {}

      self.PosValues = 0.0         # store the sum of PosDic values
      self.NegValues = 0.0         # store the sum of NegDic values
      self.PosBigramValues = 0.0   # store the sum of PosBigramDic values
      self.NegBigramValues = 0.0   # store the sum of NegBigramDic values

      if os.path.isfile("PosDic.txt") and os.path.isfile("NegDic.txt") and os.path.isfile("PosBigramDic.txt") and os.path.isfile("NegBigramDic.txt"):  # pickled files exist, load into memory
         self.PosList = self.load("PosList.txt")
         self.NegList = self.load("NegList.txt")
         self.PosDic = self.load("PosDic.txt")
         self.NegDic = self.load("NegDic.txt")
         self.PosBigramDic = self.load("PosBigramDic.txt")
         self.NegBigramDic = self.load("NegBigramDic.txt")
      else:              # pickled files do no exist, train the system
         self.train()

      # prior probability of class Positive
      self.pos_prior = float(len(self.PosList))/(len(self.PosList)+len(self.NegList))  
      print self.pos_prior
      # prior probability of class Negative
      self.neg_prior = 1 - self.pos_prior
      print self.neg_prior

      for i in xrange(len(self.PosDic.values())):
         self.PosValues += self.PosDic.values()[i]    # sum of PosDic values
      print "PosValues: " + str(self.PosValues)
      for i in xrange(len(self.NegDic.values())):
         self.NegValues += self.NegDic.values()[i]    # sum of NegDic values
      print "NegValues: " + str(self.NegValues)
      for i in xrange(len(self.PosBigramDic.values())):
         self.PosBigramValues += self.PosBigramDic.values()[i]    # sum of PosBigramDic values
      print "PosBigramValues: " + str(self.PosBigramValues)
      for i in xrange(len(self.NegBigramDic.values())):
         self.NegBigramValues += self.NegBigramDic.values()[i]    # sum of NegBigramDic values
      print "NegBigramValues: " + str(self.NegBigramValues)

   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""
      IFileList = []
      for fFileObj in os.walk("movies_reviews/"):
         IFileList = fFileObj[2]
         break

      # parse file names, categorize them into two lists: Positive and Negative
      for i in xrange(len(IFileList)):
         if IFileList[i][7] == '1':
            self.NegList.append(IFileList[i])
         elif IFileList[i][7] == '5':
            self.PosList.append(IFileList[i])
      # put into two dictionaries: Positive and Negative
      self.makeDictionary(self.PosList,self.PosDic)
      self.makeDictionary(self.NegList,self.NegDic)
      self.makeBigramDictionary(self.PosList,self.PosBigramDic)
      self.makeBigramDictionary(self.NegList,self.NegBigramDic)

      print "PosList: " + str(len(self.PosList))
      print "NegList: " + str(len(self.NegList))
      print "PosDic: " + str(len(self.PosDic))
      print "NegDic: " + str(len(self.NegDic))
      print "PosBigramDic: " + str(len(self.PosBigramDic))
      print "NegBigramDic: " + str(len(self.NegBigramDic))
      self.save(self.PosList,"PosList.txt")
      self.save(self.NegList,"NegList.txt")
      self.save(self.PosDic,"PosDic.txt")
      self.save(self.NegDic,"NegDic.txt")
      self.save(self.PosBigramDic,"PosBigramDic.txt")
      self.save(self.NegBigramDic,"NegBigramDic.txt")
      print "Successfully saved"

   def makeDictionary(self,sublist,dic):
      """
      Given a list of file names, create a dictionary to store all of the words 
      occur in total files, and the frequencies at which they occur
      Add: use LancasterStemmer to extract each word's stem before making dictionary
      """
      stemmer = nltk.stem.LancasterStemmer()   
      for i in xrange(len(sublist)):
         s = self.loadFile("movies_reviews/"+sublist[i])
         split = self.tokenize(s.lower())
         stems = [stemmer.stem(word) for word in split]   # extract each word's stem
         for j in xrange(len(stems)):
            if dic.has_key(stems[j]):   # if the key has already occured, add 1 to frequency
               dic[stems[j]] += 1
            else:
               dic_add={stems[j]:1}     # if not occured, set frequency as 1
               dic.update(dic_add)
      return dic

   def makeBigramDictionary(self,sublist,dic):
      """
      Given a list of file names, create a dictionary to store all of the words 
      occur in total files, and the frequencies at which they occur
      Difference: use bigrams to make dictionaries
      """
      for i in xrange(len(sublist)):
         s = self.loadFile("movies_reviews/"+sublist[i])
         split = self.tokenize(s.lower())
         bigrams = [(bigram[0].lower(), bigram[1].lower()) for bigram in nltk.bigrams(split)]
         for j in xrange(len(bigrams)):
            if dic.has_key(bigrams[j]):
               dic[bigrams[j]] += 1
            else:
               dic_add={bigrams[j]:1}
               dic.update(dic_add)
      return dic

   def classify(self,sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      s = self.tokenize(sText.lower())
      bigrams = [(bigram[0].lower(), bigram[1].lower()) for bigram in nltk.bigrams(s)]
      print bigrams
      stemmer = nltk.stem.LancasterStemmer()
      wordlist = [stemmer.stem(word) for word in s]
      #pos_prior = 0.0
      #neg_prior = 0.0
      pos_likelihood = 0.0
      neg_likelihood = 0.0
      pos_bigram_likelihood = 0.0
      neg_bigram_likelihood = 0.0
      p_pos = 0.0
      p_neg = 0.0
      # prior probability of class Positive
      #pos_prior = float(len(self.PosList))/(len(self.PosList)+len(self.NegList))  
      #print pos_prior
      # prior probability of class Negative
      #neg_prior = 1 - pos_prior
      #print neg_prior

      pos_likelihood = self.cal_Likelihood(self.PosDic,wordlist)
      neg_likelihood = self.cal_Likelihood(self.NegDic,wordlist)
      pos_bigram_likelihood = self.cal_Likelihood(self.PosBigramDic,bigrams)
      neg_bigram_likelihood = self.cal_Likelihood(self.NegBigramDic,bigrams)
      print pos_likelihood
      print neg_likelihood
      print pos_bigram_likelihood
      print neg_bigram_likelihood

      #p_pos = math.log(self.pos_prior)+pos_likelihood
      #p_neg = math.log(self.neg_prior)+neg_likelihood
      #p_bigram_pos = math.log(self.pos_prior)+pos_bigram_likelihood
      #p_bigram_neg = math.log(self.neg_prior)+neg_bigram_likelihood

      #distance = p_pos + p_bigram_pos - p_neg - p_bigram_neg
      #if distance > 1:
      #   return "positive"
      #elif distance < -1:
      #   return "negative"
      #else:
      #   return "neutral"
      if pos_likelihood + pos_bigram_likelihood >= neg_likelihood + neg_bigram_likelihood:
         return "positive"
      else:
         return "negative"

   def cal_Likelihood(self,dic,wordlist):
      total = 0.0
      likelihood = 0.0
      #for i in xrange(len(dic.values())):
      #   total += dic.values()[i]    # sum of Dic values
      #print total
      if(dic == self.PosDic):
         total = self.PosValues
      else:
         total = self.NegValues
      #print total
      for j in wordlist:
         if j in dic:
            likelihood += math.log((dic[j]+1)/total)
         else:
            likelihood += math.log(1/total)
      return likelihood

   #def validation(self):


   def validation_helper(self,sublist):
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
      print true_pos
      print true_neg
      print false_pos
      print false_neg
      pos_precision = float(true_pos)/float(true_pos+false_pos)
      #neg_precision = float(false_pos)/float(true_pos+false_pos)
      pos_recall = float(true_pos)/float(true_pos+false_neg)
      #neg_recall = float(true_neg)/float(true_neg+false_pos)
      pos_f_measure = 2*pos_precision*pos_recall/(pos_precision+pos_recall)
      #neg_f_measure = 2*neg_precision*neg_recall/(neg_precision+neg_recall)
      print pos_precision
      print pos_recall
      print pos_f_measure 
      #print neg_precision
      #print neg_recall
      #print neg_f_measure 


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
      for i in xrange(fold):
         sublist.append([])
      for j in xrange(len(corpus)):
         sublist[j%fold].append(corpus[j])
      return sublist

a = Bayes_Classifier()
IFileList = []
for fFileObj in os.walk("movies_reviews/"):
   IFileList = fFileObj[2]
   break
result = segment(IFileList,10)
a.validation_helper(result[0])











