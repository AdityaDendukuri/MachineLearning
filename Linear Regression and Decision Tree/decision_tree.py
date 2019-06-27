import random

class person:
      def __init__(self, id, wine, running, pizza):
            self.id = id
            self.likes_wine = wine
            self.likes_running = running
            self.likes_pizza = pizza
            self.likes_beer = False
      def get_attribute(self, i):
            if i == 0:
                  return self.likes_wine
            if i == 1:
                  return self.likes_running
            if i == 2:
                  return self.likes_pizza
                  
all_students = []
like_pizza = []
like_running = []
like_wine = []

random.seed()

wine_id = random.sample(xrange(0,99), 50)
running_id = random.sample(xrange(0,99), 30)
pizza_id = random.sample(xrange(0,99), 80)

for i in range(100):
      all_students.append(person(i, False, False, False))

for i in range(len(wine_id)):
      all_students[wine_id[i]].likes_wine = True
      like_wine.append(all_students[wine_id[i]])
for i in range(len(running_id)):
      all_students[running_id[i]].likes_running = True
      like_running.append(all_students[running_id[i]])
for i in range(len(pizza_id)):
      all_students[pizza_id[i]].likes_pizza = True
      like_pizza.append(all_students[pizza_id[i]])


def I(D, attribute):
      p_plus = 0.0
      p_minus = 0.0
      for i in range(len(D)):
            if(D[i].get_attribute(attribute)):
                  p_plus += 1.0
            else:
                  p_minus += 1.0
      p_plus /= float(len(D))
      p_minus /= float(len(D))
      return len(D) * (1 - p_plus*p_plus - p_minus*p_minus)


random.shuffle(all_students)
all_students_left = all_students[0:int(len(all_students)/2)]
all_students_right = all_students[int(len(all_students)/2):]


I_full_wine = I(all_students, 0)
I_left_wine = I(all_students_left, 0)
I_right_wine = I(all_students_right, 0)

I_full_running = I(all_students, 1)
I_left_running = I(all_students_left, 1)
I_right_running = I(all_students_right, 1)

I_full_pizza = I(all_students, 2)
I_left_pizza = I(all_students_left, 2)
I_right_pizza = I(all_students_right, 2)


gini_wine = I_full_wine - I_left_wine - I_right_wine
gini_running = I_full_running - I_left_running - I_right_running
gini_pizza = I_full_pizza - I_left_pizza - I_right_pizza


print("GINI wine: ", gini_wine)
print("GINI running: ", gini_running)
print("GINI pizza: ", gini_pizza)
