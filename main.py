import math
import random
import numpy as np

num_city = 100
num_air = 3
num_center = 5
sigma = 0.1
cities = set()
airports = []

for i in range(num_center):
    x = random.random()
    y = random.random()
    xc = np.random.normal(x, sigma, num_city//num_center)
    yc = np.random.normal(y, sigma, num_city//num_center)
    cities = cities.union(zip(xc, yc))


for i in range(num_air):
    x = random.random()
    y = random.random()
    airports.append((x,y))

import matplotlib.pyplot as plt

zip_cities = zip(*cities)
plt.scatter(*zip_cities, marker='+',color='b', label='Cities')
zip_airs = zip(*airports)
plt.scatter(*zip_airs, marker='*', color='r', s=100, label='Airports')
plt.legend()
plt

def distance(airport, city):
    R = 3958.8
    x = airport[0]
    y = airport[1]
    xc = city[0]
    yc = city[1]
    dist = 2 * R * math.asin(math.sqrt((math.sin(abs(xc - x) / 2)) ** 2 + (math.cos(x) * math.cos(xc) * (math.sin(abs(yc - y) / 2)) ** 2)))

    return dist


def citySetCal(airports, cities):
    citySets = []
    for i in range (len(airports)):
        citySets.append([])
    for city in cities:
        closestIndex = 0
        minDistance = distance(airports[0], city)
        for i, airport in enumerate(airports[1:]):
            current = distance(airport, city)
            if current < minDistance:
                closestIndex = i+1
                minDistance = current
        citySets[closestIndex].append(city)

    return citySets

def objectiveFunction(airports,citySets):
    sumC = 0
    sumT = 0
    i = 0

    for cities in citySets:
        airport = airports[i]
        x = airport[0]
        y = airport[1]
        for city in cities:
            xC = city[0]
            yC = city[1]
            sumC += ((x - xC) ** 2)+ ((y - yC) ** 2)
        i += 1
        sumT += sumC

    return sumT



def calculateGradient(airports,citySets):

    #[df/dx1,df/y1....]
    gradient =[]
    #helps update the airports list from 1-3
    airportsIndex = 0
    # will store the gradients
    dfdx =0
    dfdy =0
    #Summation df/dx (xi - xc)^2
    for citySet in citySets:
        airport = airports[airportsIndex]
        for city in citySet:
            dfdx += 2 * airport[0]-city[0]
            dfdy += 2 * airport[1]-city[1]
        gradient.append(dfdx)
        gradient.append(dfdy)
        dfdx = 0
        dfdy = 0
        airportsIndex +=1

    return gradient

def gradientDecent(airports, cities):
    state =[]
    objectiveValues=[]
    for k in range (len(airports)):
        state.append([])
    j = 0
    for airport in airports:
        state[j].append(airport[0])
        state[j].append(airport[1])
        j += 1
    print ("State")
    print (state)
    delta = 1
    percentChange = 0.00001

    citySets = citySetCal(airports, cities)

    print("CitySets")
    print(citySets)
    objectiveValues.append(objectiveFunction(state,citySets))

    while percentChange > delta:
        # Computing gradient
        gradientVector = calculateGradient(airports, citySets)
        alpha = 0.000000001
        index = 0
        # updating and moving current state
        for curr in state:
            state[index] = curr - alpha * gradientVector[index]
            index += 1

        # Calculating new city sets
        citySets = citySetCal(airports, cities)

        # Store the objective function
        objectiveValues.append(objectiveFunction(state, citySets))
        currentValue = objectiveValues[-1]
        previousValue = objectiveValues[-2]
        percentChange = abs((currentValue-previousValue)/previousValue) #check
        print("%Change", percentChange)

    return state, objectiveValues

if __name__ == '__main__':
    citySets = citySetCal(airports, cities)
    #print(citySets)
    #objectiveFunction(airports, citySets)
    state, objectiveValues = gradientDecent(airports, cities)
    print("obj")
    print(objectiveValues)