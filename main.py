##classification with brand and subcategory

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm #deferent models were tested for better prediction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer	#data prepocessing requirement
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


dirname, filename = os.path.split(os.path.abspath(__file__))



#main path to the data folder
path_to_data = dirname+'/data/'



#-------------------------------------------------------------------------------
#	this section is for the classification of the category1 the main category


# primary category data fetched

# text and labels as X and y
df = pd.read_csv(path_to_data+'dataset.csv')
X = df['name']
y = df['category1']


# data cleaning operations (lowering the letters)
xdf = X.str.lower()
X = xdf

# data cleaning operations (removing stop words)
from nltk.corpus import stopwords
stop = stopwords.words('english')
X = X.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# creating training and cross_validating data
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, y)

# encoding the lables
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)



# data preprocessing - transform the training and validation data using count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(X)
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)



#model training and prediction
classifier = linear_model.LogisticRegression(solver='liblinear')
classifier.fit(xtrain_count, train_y)
predictions = classifier.predict(xvalid_count)
accuracy = metrics.accuracy_score(predictions, valid_y)
# comment below to hide the accuracy of the main classifier 
print("primary category classification accuracy : " , "{0:.4f}".format(accuracy) ) 


#----------------------------------------------------------------------------------------

# method for predicting the categories for an input 'test'
def run_test(test,test_classifier,count_vect,test_encoder):
    test_count =  count_vect.transform(test)
    pred = test_classifier.predict(test_count)
    pred_proba = test_classifier.predict_proba(test_count)
    pred = test_encoder.inverse_transform(pred)
    #print(pred_proba[0])
    return pred, max(pred_proba[0])


# ---------------------------------------------------------------------------------------
	# this section contain the classification of category2


def get_sub_classifier(filename,isbrand = False):	#creating a classifier for a category or a brand
	file = path_to_data+'sub_category/'+filename
	df = pd.read_csv(file)
	X = df['name']
	y = df['category2']
	if (isbrand==True):
		y = df['brand']

	xdf = X.str.lower()
	X = xdf

	from nltk.corpus import stopwords
	stop = stopwords.words('english')
	X = X.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

	train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, y)

	sub_encoder = preprocessing.LabelEncoder()
	y = sub_encoder.fit_transform(y)

	train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X, y)

	count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
	count_vect.fit(X)
	xtrain_count =  count_vect.transform(train_x)
	xvalid_count =  count_vect.transform(valid_x)

	classifier = linear_model.LogisticRegression(solver='liblinear')
	classifier.fit(xtrain_count, train_y)
	predictions = classifier.predict(xvalid_count)
	accuracy = metrics.accuracy_score(predictions, valid_y)
	#comment to hide the accuracy of each subclassifier
	if(isbrand==True):
		print("Brand Accuracy for subcategory (",filename,") : ","{0:.4f}".format(accuracy))
	else:
		print("Category Accuracy for subcategory (",filename,") : ","{0:.4f}".format(accuracy))
	
	return classifier,count_vect, sub_encoder




#--------------------------------------------------------------------------------------



# create classifier for each category1 elements
home_classifier,count_vect1 ,sub_encoder1 = get_sub_classifier('Home & Kitchen.csv')
fashion_classifier,count_vect2 ,sub_encoder2= get_sub_classifier('Fashion Accessories.csv')
electronics_classifier,count_vect3,sub_encoder3  = get_sub_classifier('Electronics.csv')
footware_classifier,count_vect4,sub_encoder4  = get_sub_classifier('Footware.csv')
clothing_classifier,count_vect5 ,sub_encoder5 = get_sub_classifier('Clothing.csv')


#-------------------------------------------------------------------------------------
	# brand categorisation

# create classifier for each brand category

brand_home_classifier,brand_count_vect1 ,brand_sub_encoder1 = get_sub_classifier('Home & Kitchen.csv',isbrand = True)
brand_fashion_classifier,brand_count_vect2 ,brand_sub_encoder2= get_sub_classifier('Fashion Accessories.csv',isbrand = True)
brand_electronics_classifier,brand_count_vect3,brand_sub_encoder3  = get_sub_classifier('Electronics.csv',isbrand = True)
brand_footware_classifier,brand_count_vect4,brand_sub_encoder4  = get_sub_classifier('Footware.csv',isbrand = True)
brand_clothing_classifier,brand_count_vect5 ,brand_sub_encoder5 = get_sub_classifier('Clothing.csv',isbrand = True)


# method execute and print the details for the given input
def get_details(input):

	pred1, prob1 = run_test(input,classifier,count_vect,encoder)

	category = pred1[0]

	# according to the output from the main category decide which sub category will be followed
	# so make a parameters according to that category

	if category == 'Electronics':
	    sub_classifier = electronics_classifier
	    sub_count_vect = count_vect3
	    sub_encoder = sub_encoder3
	    brand_sub_classifier = brand_electronics_classifier	# for brand classification
	    brand_sub_count_vect = brand_count_vect3			# for brand classification
	    brand_sub_encoder = brand_sub_encoder3				# for brand classification
	elif category == 'Home & Kitchen & Automotive':
	    sub_classifier = home_classifier
	    sub_count_vect = count_vect1
	    sub_encoder = sub_encoder1
	    brand_sub_classifier = brand_home_classifier
	    brand_sub_count_vect = brand_count_vect1
	    brand_sub_encoder = brand_sub_encoder1
	elif category=='Clothing':
	    sub_classifier = clothing_classifier
	    sub_count_vect = count_vect5
	    sub_encoder = sub_encoder5
	    brand_sub_classifier = brand_clothing_classifier
	    brand_sub_count_vect = brand_count_vect5
	    brand_sub_encoder = brand_sub_encoder5
	elif category=='footware':
	    sub_classifier = footware_classifier
	    sub_count_vect = count_vect4
	    sub_encoder = sub_encoder4
	    brand_sub_classifier = brand_footware_classifier
	    brand_sub_count_vect = brand_count_vect4
	    brand_sub_encoder = brand_sub_encoder4
	elif category=='Fashion Accesories':
	    sub_classifier = fashion_classifier
	    sub_count_vect = count_vect2
	    sub_encoder = sub_encoder2
	    brand_sub_classifier = brand_fashion_classifier
	    brand_sub_count_vect = brand_count_vect2
	    brand_sub_encoder = brand_sub_encoder2

	pred2, prob2 = run_test(input,sub_classifier,sub_count_vect,sub_encoder)

	brand_pred, brand_prob = run_test(input,brand_sub_classifier,brand_sub_count_vect,brand_sub_encoder)


	print(input[0])
	print("{\ncategory_tree", " : ",pred1[0]," >> ",pred2[0])
	print("brand : ",brand_pred[0])
	print("scores : [ ")
	print("\t","category : ",pred1[0],", confidence"," : ","{0:.2f}".format(prob1))
	print("\t","category : ",pred2[0],", confidence"," : ","{0:.2f}".format(prob2))
	print("\t","brand : ",brand_pred[0],", confidence"," : ","{0:.2f}".format(brand_prob))
	print("\t]")
	print("}")
	print("\n")

#--------------------------------------------------------------------------------------



#comment below to take input from the terminal.

path_to_assignment = dirname
f = open(path_to_assignment+'/input.txt', 'r')	#taking input from 'input.txt' file
lines = f.readlines()
for line in lines:
	get_details([line])


# for taking input from console
#print("Enter input : ")
#while(1):
#	s = str(input())
#	if(s=="exit"):
#		break
#	get_details([s])
      
