users =[
    { "id":0, "name": "Hero" },
    { "id":1, "name": "Dunn" },
    { "id":2, "name": "Sue" },
    { "id":3, "name": "Chi" },
    { "id":4, "name": "Thor" },
    { "id":5, "name": "Clive" },
    { "id":6, "name": "Hicks" },
    { "id":7, "name": "Devin" },
    { "id":8, "name": "Kate" },
    { "id":9, "name": "Klein" }    
      ]

friendship = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (6, 8),
    (7, 8),
    (8, 9)
]

def num_friends(user):
    # find number of friends for a given user
        return len(user["friends_names"])

for user in users:
    user["friends_names"] = [] 

   
    
    # Populating the friends names
for i,j in friendship:
    users[i]["friends_names"].append(users[j]) 
    users[j]["friends_names"].append(users[i]) 


for user in users:
    print(user["name"],"has", num_friends(user),"friends")



listby_id = [(user["name"],num_friends(user))for user in users]



val = sorted(listby_id, key=lambda number:number[1] , reverse = True)

print("Sorting user list from most number of friends to least number")
print(val)

