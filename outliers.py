
import matplotlib.pyplot as plt 
import csv 
import pandas as pd 
from sklearn.ensemble import IsolationForest
  
#barriers = [] 
#distance = [] 
  
#with open('data/vaskas_features_properties_smiles_filenames.csv','r') as csvfile: 
#    lines = csv.reader(csvfile, delimiter=',')
#    next(lines) 
#    for row in lines: 
#        distance.append(float(row[90])) 
#        barriers.append(float(row[91])) 
  
#plt.scatter(distance, barriers, color = 'g',s = 50)  
#plt.xlabel('Distance') 
#plt.ylabel('Barrier') 
#plt.title('Distance vs. Barrier Hight', fontsize = 20) 
  
#plt.show() 

distance_barrier = pd.read_csv("data/vaskas_features_properties_smiles_filenames.csv", usecols=[90, 91])
print(distance_barrier.head())
distance_barrier.plot(kind='scatter', x='barrier', y='distance')
plt.show() 

clf = IsolationForest()
clf.fit(distance_barrier)

#from sklearn.inspection import DecisionBoundaryDisplay

#disp = DecisionBoundaryDisplay.from_estimator(
#    clf,
#    distance_barrier,
#    response_method="predict",
#    alpha=0.5,
#)
#disp.ax_.scatter(distance_barrier)
#plt.axis("square")
#plt.show()