from sklearn.feature_extraction.text import CountVectorizer
import pickle

classifier_filename = 'final_classifier.pkl'
vectorizer_filename = 'final_vectorizer.pkl'

loaded_model = pickle.load(open(classifier_filename, 'rb'))
loaded_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

testList = list()
#testList.append("this is bullshit")
#testList.append("this is boring")
#testList.append("this is stupid")
#testList.append("this is great")
#testList.append("this is awful, the worst product I bought")
#testList.append("this is not that good")
#testList.append("this is not that good, I am disappointed")
#testList.append("this is not good")
#testList.append("this is not bad")
#testList.append("this is what I call a total fail")
#testList.append("this is what I call a great fail")
#testList.append("I would not recommend")
#testList.append("I would not reccomend")
#testList.append("I recommend")
#testList.append("I reccomend")
#testList.append("Our second Acer, loved the first for almost 3 years. I will never do business with them again!")
#testList.append("I will never do business with them again!")
#testList.append("Can't deal with the noise!")
file=open("input.txt",'r')
for line in file:
    testList.append(line)


testJulien = loaded_vectorizer.transform(testList)

#print(loaded_model.predict(testJulien))
ans=loaded_model.predict(testJulien)
print("\nThe results are");
for i in range(len(testList)):
    if(ans[i]==1):
        print("\nPositive")
    else:
         print("\nNegative")
    print("\n"+testList[i])        