import numpy as np
import matplotlib.pyplot as plt 
 
  
# creating the dataset
data = {'Base Phi-2':0, 'Finetuned Phi-2':198, 'Tie':2}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color =([0/256, 39/256, 76/256, 1], [255/256, 203/256, 5/256, 1], [0/256, 0/256, 0/256, 1]), 
        width = 0.4)
 
plt.xlabel("Preferred LLM Model")
plt.ylabel("Number of Preferred Answers")
plt.title("Preferred Code Summarization Generated Answers With Chat GPT 3.5 Turbo as a Judge")
plt.show()