import star_wars_textgeneration

#the dot is look into the same directory as the folder
#Getting the information of the 1st Star Wars Movie

path_to_starwarsIV = "./SW_EpisodeIV.txt"
with open(path_to_starwarsIV, "r") as f:
    starwarIV = f.read()
print(str(len(starwarIV)) + " character(s)")
chars = set(starwarIV)
print("'~' is a good pad character: ", "~" not in chars)

#Getting the data for the 2nd Star Wars Movie
path_to_starwarsV = "./SW_EpisodeVI.txt"
with open(path_to_starwarsV, "r") as f:
    starwarV = f.read()
print(str(len(starwarV)) + " character(s)")
chars = set(starwarV)
print("'~' is a good pad character: ", "~" not in chars)


#Getting the data for the e3rd Star Wars Movie
path_to_starwarsVI = "./SW_EpisodeVI.txt"
with open(path_to_starwarsVI, "r") as f:
    starwarVI = f.read()
print(str(len(starwarVI)) + " character(s)")
chars = set(starwarVI)
print("'~' is a good pad character: ", "~" not in chars)

starwarIV = starwarIV[23:]
starwarIV = ''.join([i for i in starwarIV[:] if not i.isdigit()])
starwarIV = starwarIV.replace('""', "")
#print(starwarIV[:1000])

starwarV = starwarV[23:]
starwarV = ''.join([i for i in starwarV[:] if not i.isdigit()])
starwarV = starwarV.replace('""', "")

starwarVI = starwarVI[23:]
starwarVI = ''.join([i for i in starwarVI[:] if not i.isdigit()])
starwarVI = starwarVI.replace('""', "")

all_3starwars = starwarIV + starwarV + starwarVI

"""
generates the actual text that is to be randomized.
"""
t0 = time.time()
lm3 = train_lm(all_3starwars, 11)

print(generate_text(lm3, 11, "Hello there", 2000))