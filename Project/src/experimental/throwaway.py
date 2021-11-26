import matplotlib.pyplot as plt

file = open(r'C:\Users\Test\Documents\School\2021-H\IDATT2502\Course Work\Project\Documentation\A4_v4_3 log.txt', 'r')
lines = file.readlines()

rewards = []
for line in lines:
    index = line.find("with ")
    line = line[index+5:]
    index = line.find(" ")
    line = line[:index]
    index = line.find(",")
    if index != -1:
        line = line[:index]
    print(line)
    try:
        rew = float(line)
    except:
        continue
    rewards.append(float(line))

plt.plot(rewards)
plt.title(f"Reward over {len(rewards)} episodes")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
