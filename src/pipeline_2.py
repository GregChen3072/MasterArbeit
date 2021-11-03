# Importing required libraries
import pandas as pd

record = {
    'course_name': ['Data Structures', 'Python',
                    'Machine Learning', 'Web Development'],
    'student_name': ['Ankit', 'Shivangi',
                     'Priya', 'Shaurya'],
    'student_city': ['Chennai', 'Pune',
                     'Delhi', 'Mumbai'],
    'student_gender': ['M', 'F',
                       'F', 'M']}

# Creating a dataframe
df = pd.DataFrame(record)

# Creating a dataframe with 75%
# values of original dataframe
part_75 = df.sample(frac=0.75)

# Creating dataframe with
# rest of the 25% values
rest_part_25 = df.drop(part_75.index)

print("\n75% of the given DataFrame:")
print(part_75)

print("\nrest 25% of the given DataFrame:")
print(rest_part_25)
