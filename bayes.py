# Name: 
# Date:
# Description:
#
#

import math, os, pickle, re

class Bayes_Classifier:

   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""
      self.PosList = []
      self.NegList = []
      self.PosDic = {}
      self.NegDic = {}
      if os.path.isfile("PosDic.txt") and os.path.isfile("NegDic.txt"):  # pickled files exist, load into memory
         self.PosDic = self.load("PosDic.txt")
         self.NegDic = self.load("NegDic.txt")
      else:              # pickled files do no exist, train the system
         self.train()

   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""
      IFileList = []
      for fFileObj in os.walk("movies_reviews/"):
         IFileList = fFileObj[2]
         break

      # parse file names, categorize them into two lists: Positive and Negative
      for i in xrange(len(IFileList)):
         if IFileList[i][7] == '1':
            self.PosList.append(IFileList[i])
         elif IFileList[i][7] == '5':
            self.NegList.append(IFileList[i])
      # put into two dictionaries: Positive and Negative
      self.makeDictionary(self.PosList,self.PosDic)
      self.makeDictionary(self.NegList,self.NegDic)

      print "PosList: " + str(len(a.PosList))
      print "NegList: " + str(len(a.NegList))
      print "PosDic: " + str(len(a.PosDic))
      print "NegDic: " + str(len(a.NegDic))
      self.save(self.PosDic,"PosDic.txt")
      self.save(self.PosDic,"NegDic.txt")
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
      print wordlist
      pos_prior = 0.0
      neg_prior = 0.0
      pos_likelihood = 0.0
      neg_likelihood = 0.0
      p_pos = 0.0
      p_neg = 0.0
      # prior probability of class Positive
      pos_prior = float(len(self.PosList))/(len(self.PosList)+len(self.NegList))  
      print pos_prior
      # prior probability of class Negative
      neg_prior = 1 - pos_prior
      print neg_prior

      pos_likelihood = self.cal_Likelihood(self.PosDic,wordlist)
      neg_likelihood = self.cal_Likelihood(self.NegDic,wordlist)
      print pos_likelihood
      print neg_likelihood

      p_pos = pos_prior*pos_likelihood
      p_neg = neg_prior*neg_likelihood

      distance = p_pos - p_neg
      if distance > 1:
         return "positive"
      elif distance < -1:
         return "negative"
      else:
         return "neutral"

   def cal_Likelihood(self,dic,wordlist):
      total = 0.0
      likelihood = 0.0
      for i in xrange(len(dic.values())):
         total += dic.values()[i]    # sum of PosDic values
      print total
      for j in wordlist:
         if j in dic:
            likelihood += math.log((dic[j]+1)/total)
         else:
            likelihood += math.log(1/total)
      return likelihood

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
